import copy
import inspect
import json
import math
import os
import pickle
from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod
from collections import OrderedDict
from dataclasses import field
import socket
from datetime import datetime
from threading import Thread
from typing import Callable, Dict, Any, Union, List, Optional, Iterable, Tuple
import datasets
import nltk
import numpy as np
import torch
from datasets import Dataset
from ray import tune
from transformers.trainer_utils import PredictionOutput, EvalPrediction
from transformers.utils import ModelOutput

from .loss import LOSS_FN_NAME2CLASS
from ..argument import ArgumentConfigurator, ExternalArgument, ArgumentPool, argument_class, ArgumentParser, \
    NULL_ARGUMENT, Argument, FieldConflictError
from ..data import Data, DatasetSplitType
from ..data.data import TaskType, Name2DataClass
from ..pipeline import UniversalPort
from .approach import NeuralNetWork, NeuralNetWorkArgument, NNModelArgument
from transformers import Trainer, PreTrainedModel, TrainingArguments, DefaultDataCollator, \
    PretrainedConfig, PreTrainedTokenizer, TrainerCallback, TrainerState, TrainerControl

from ..utils.utils import MetaMetrics, SequencePairBatch, update_instant_attribute, refer_forward

EXTEND_MODULE_NAME = 'extend_module'


@argument_class
class TransformerArgument(NeuralNetWorkArgument):
    pass


class ExtendModuleForm:
    def __init__(self, model: torch.nn.Module, pre_plm_forward: Callable = None, post_plm_forward: Callable = None):
        if pre_plm_forward is None and post_plm_forward is None:
            raise ValueError
        self._model = model
        self._pre_plm_forward = pre_plm_forward
        self._post_plm_forward = post_plm_forward

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def pre_plm_forward(self):
        return self._pre_plm_forward

    @property
    def post_plm_forward(self):
        return self._post_plm_forward


def insert_call_back_to_forward(model: torch.nn.Module = None, pre_forward_call_back: Callable = None, post_forward_call_back: Callable = None):
    if not isinstance(model, torch.nn.Module):
        return False

    def forward_wrapped_call_back(model_, *args, **kwargs):
        result = {}

        def call_pipeline(call):
            call_signature = inspect.signature(call).parameters
            need_kwargs = {k: v for k, v in kwargs.items() if k in call_signature}
            return call(**need_kwargs)

        if isinstance(pre_forward_call_back, Callable):
            o = call_pipeline(pre_forward_call_back)
            if isinstance(o, dict):
                kwargs.update(o)

        result = call_pipeline(getattr(model_, 'override_forward'))

        if isinstance(post_forward_call_back, Callable):
            kwargs.update(result)
            result = call_pipeline(post_forward_call_back)
        assert isinstance(result, ModelOutput) and 'logits' in result
        return result

    signature_parameters = OrderedDict(inspect.signature(model.forward).parameters)
    if isinstance(pre_forward_call_back, Callable):
        signature_parameters.update(inspect.signature(pre_forward_call_back).parameters)
    if isinstance(post_forward_call_back, Callable):
        signature_parameters.update(inspect.signature(post_forward_call_back).parameters)
    refer_forward(model, forward_wrapped_call_back)
    model.forward.__signature__ = inspect.Signature(parameters=list(signature_parameters.values()))
    return True


@argument_class
class TransformerModelArgument(NNModelArgument):
    plm_name_or_path: str = field(
        metadata={"help": "Path to local pretrained model or model identifier from huggingface.co/models"}
    )
    plm_config_name: str = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "The path to save pretrained models downloaded from huggingface.co. "
                          "(origin: transformers)"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific version of model to use (branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    # @property
    # def model_list(self):
    #     result = [mode_info.id for mode_info in self._model_list]
    #     return result

    def __post_init__(self):
        super(TransformerModelArgument, self).__post_init__()
        from huggingface_hub import list_models
        hf_model_list_cache = os.path.join(ArgumentPool.meta_arguments.cache_dir, 'hf_model_list.pkl')
        if not os.path.isfile(hf_model_list_cache) or ArgumentPool.meta_arguments.force_cache:

            model_id2info = {m.modelId: m for m in list_models()}
            with open(hf_model_list_cache, mode='wb') as f:
                pickle.dump(model_id2info, f)
        else:
            with open(hf_model_list_cache, mode='rb') as f:
                model_id2info = pickle.load(f)

        name = None
        if not os.path.isdir(self.plm_name_or_path):
            candidates = self.plm_name_or_path.split('/')
            for c in candidates:
                if list_models(search=c):
                    if not name is None:
                        raise ValueError
                    name = c
            self._plm_name = None
            self._model_name = None
        else:
            name = self.plm_name_or_path
            self._plm_name = name
            self._model_name = name

        if name is None:
            print(f"{self.plm_name_or_path} is perhaps not exist on huggingface hub")
            raise ValueError


    @property
    def plm_name(self) -> str:
        return self._plm_name


class Transformer(NeuralNetWork, ABC):
    _type = 'transformer'
    _argument_class = TransformerArgument
    _model_argument_class = TransformerModelArgument
    _training_arg_class: type = TrainingArguments
    _trainer_class = Trainer
    _data_collator_class: type = DefaultDataCollator

    def __init__(self, *args, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)
        self._data_collator = None
        self._tokenizer = self._tokenizer_init(**kwargs)
        self._extend_module_info: ExtendModuleForm = None
        self._plm: PreTrainedModel = None

    def release_model(self, model=None):
        super().release_model(model)
        if self._plm is not None:
            self._plm.to_empty(device=torch.device('cpu'))

    def _update_args(self):
        super(Transformer, self)._update_args()
        self.args: TransformerArgument
        if self.args.tune_hp:
            self.training_args.do_train = \
                self.training_args.do_eval = True

        base_dir = self.args.output_dir
        self.training_args.output_dir = os.path.join(base_dir, 'model')
        self.training_args.save_model_path = os.path.join(self.training_args.output_dir, 'trained')
        self.training_args.logging_dir = os.path.join(base_dir, 'log')
        self.training_args.tune_hp_dir = os.path.join(base_dir, 'ray', 'tune')
        self.training_args.include_inputs_for_metrics = True
        self.training_args.eval_accumulation_steps = 128 // self.training_args.per_device_eval_batch_size
        if self.args.tune_hp:
            self.training_args.evaluation_strategy = "epoch"
            self.training_args.use_mps_device = False

        if not isinstance(self.args.loss_fn, str):
            self.args.loss_fn = self.loss_fn_class
        else:
            self.args.loss_fn = LOSS_FN_NAME2CLASS[self.args.loss_fn]

    @property
    def loss_fn_class(self) -> Callable:
        return None

    @property
    @abstractmethod
    def auto_classes(self):
        ...

    @property
    def check_point(self):
        checkpoint = self.model_args.plm_name_or_path
        if not self.training_args.do_train and not self.args.tune_hp and os.path.isdir(self.training_args.save_model_path):
            checkpoint = self.training_args.save_model_path

        return checkpoint

    def _config_init(self, data: Data = None,  check_point:str = None, auto_classes: type = None, **kwargs):
        processing_data = data if isinstance(data, Data) else self.processing_data
        if processing_data.task_type == TaskType.CLASSIFICATION:
            kwargs["num_labels"] = len(processing_data.category_names)
        elif processing_data.task_type == TaskType.GENERATION:
            kwargs["max_length"] = processing_data.max_length \
                if hasattr(processing_data, "max_length") and isinstance(processing_data.max_length, int) else 128
        elif processing_data.task_type == TaskType.REGRESSIVE:
            kwargs["num_labels"] = 1
        else:
            raise ValueError

        config = auto_classes.from_pretrained(check_point, **kwargs)

        return config

    def plm_is_from_check_point(self):
        return os.path.isdir(self.model_args.plm_name_or_path)

    def _plm_init(self, check_point:str = None, init_tokenizer=False, auto_classes:Dict[str, type] = None, data: Data = None, **kwargs):

        auto_classes = auto_classes if isinstance(auto_classes, dict) else self.auto_classes
        processing_data = data if isinstance(data, Data) else self.processing_data
        if init_tokenizer:
            self._tokenizer = self._tokenizer_init(check_point, auto_classes)

        config = self._config_init(processing_data, check_point, auto_classes['config'], **kwargs)

        print(f"***********load model for task {processing_data.abbreviation} from {check_point}**********")



        plm = auto_classes['model'].from_pretrained(check_point, config=config)
        self._plm = plm
        return plm

    @property
    def plm(self) -> PreTrainedModel:
        return self._plm

    def _extend_module_init(self, model, extend_module_form: ExtendModuleForm, check_point:str = None, **kwargs) -> Optional[Union[torch.nn.Module,torch.nn.ModuleDict]]:
        if isinstance(extend_module_form, ExtendModuleForm):
            self._extend_module_info = extend_module_form
            assert isinstance(extend_module_form.model, torch.nn.Module) and \
                   (isinstance(extend_module_form.pre_plm_forward, Callable) or
                    isinstance(extend_module_form.post_plm_forward, Callable))

            if check_point is not None and os.path.isfile(os.path.join(check_point, f'{EXTEND_MODULE_NAME}.bin')):
                archive_path = os.path.join(check_point, f'{EXTEND_MODULE_NAME}.bin')
                extend_module_form.model.load_state_dict(torch.load(archive_path))

            setattr(model, EXTEND_MODULE_NAME, extend_module_form.model)

            assert insert_call_back_to_forward(model, pre_forward_call_back=extend_module_form.pre_plm_forward,
                                               post_forward_call_back=extend_module_form.post_plm_forward)
        return model

    @property
    def extend_module_info(self) -> Optional[ExtendModuleForm]:
        return self._extend_module_info

    def _model_init(self, check_point:str = None, init_tokenizer=False, auto_classes:Dict[str, type] = None, data: Data = None, extend_module_form: ExtendModuleForm = None, **kwargs) -> Optional[PreTrainedModel]:
        check_point = check_point if isinstance(check_point, str) else self.check_point
        model = self._plm_init(check_point=check_point, init_tokenizer=init_tokenizer,
                               auto_classes=auto_classes, data=data, **kwargs)
        extend_module_form = extend_module_form if isinstance(extend_module_form,
                                                              ExtendModuleForm) else self.extend_module_info
        self._extend_module_init(model, extend_module_form, check_point)

        if self.training_args.do_train:
            model.train()
        else:
            model.eval()

        self._model = model

        return model

    def _tokenizer_init(self, check_point:str = None, auto_classes:Dict[str, type] = None):
        check_point = check_point if isinstance(check_point, str) else self.check_point
        auto_classes = auto_classes if isinstance(auto_classes, dict) else self.auto_classes
        print(f"***********load tokenizer of transformer {self.model_args.model_name} from {check_point}**********")
        return auto_classes['tokenizer'].from_pretrained(check_point)

    def pre_tokenization_call_back(self, sequence):
        return sequence

    def _tokenize(self, original_inputs, original_labels=None, prefix=None, max_length=128, truncation=True, **kwargs):
        tokenizer: PreTrainedTokenizer = self.tokenizer
        if isinstance(original_inputs, SequencePairBatch):
            original_inputs = original_inputs.first_sequence_batch, original_inputs.second_sequence_batch
        elif isinstance(original_inputs, Tuple):
            pass
        elif isinstance(original_inputs[0], List) and isinstance(original_inputs[0][0], str):
            original_inputs = np.array(original_inputs).transpose().tolist()
        elif isinstance(original_inputs[0], str):
            original_inputs = original_inputs,
        else:
            raise ValueError

        original_inputs = [self.pre_tokenization_call_back(o) for o in original_inputs]
        _inputs = tokenizer(*original_inputs, max_length=max_length, truncation=truncation, **kwargs)
        if original_labels is not None:
            data = self.processing_data
            if data.task_type in (TaskType.CLASSIFICATION, TaskType.REGRESSIVE):
                _inputs[data.label_name] = original_labels
            elif data.task_type == TaskType.GENERATION:
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(original_labels, max_length=max_length, truncation=truncation, **kwargs)
                _inputs[data.label_name] = labels["input_ids"]
            else:
                raise ValueError
        if prefix is not None:
            _inputs = {f"{prefix}_{k}": v for k, v in _inputs.items()}
        return _inputs

    @classmethod
    @property
    def training_arg_class(cls):
        return cls._training_arg_class

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            raise ValueError
        return self._tokenizer

    @property
    def data_collator(self):
        class_call_signature = set(inspect.signature(self._data_collator_class).parameters.keys())
        kwargs = {}
        if "tokenizer" in class_call_signature:
            kwargs["tokenizer"] = self.tokenizer
        if "model" in class_call_signature:
            kwargs["model"] = self.model
        if self._data_collator is None:
            self._data_collator = self._data_collator_class(**kwargs)
        return self._data_collator

    def _pre_call_hock(self):
        super(Transformer, self)._pre_call_hock()
        data_splits = set()
        if self.training_args.do_train:
            data_splits.add(DatasetSplitType.TRAIN)
        if self.training_args.do_eval:
            data_splits.add(DatasetSplitType.VALIDATION)
        if self.training_args.do_predict:
            data_splits.add(DatasetSplitType.TEST)
        self._process_data_split = tuple(data_splits)
        if not isinstance(self.training_args.label_names, str):
            self.training_args.label_names = [self.processing_data.label_name]

    @classmethod
    def collect_argument(cls, *arg, **kwargs):
        super(Transformer, cls).collect_argument()
        if cls.training_arg_class is None:
            raise ValueError

        cls.training_arg_class = argument_class(cls.training_arg_class, bases=NULL_ARGUMENT)
        try:
            ArgumentPool.push(cls.training_arg_class)
        except FieldConflictError:
            cls.training_arg_class = ArgumentPool.push(cls.training_arg_class, revise_cls = True)

    def assign_argument(self):
        super().assign_argument()
        self.training_args: TrainingArguments = ArgumentPool.pop(self.training_arg_class)
        # self.training_args.approach_args = self.args
        pass

    def _preprocess(self, samples: Dataset, *args, **kwargs) -> Any:
        model_inputs = self._tokenize(*self.precessing_data.extract_input_label_from_samples(samples))
        return model_inputs

    def _save_trained_model(self, trainer: Trainer, save_model_path:str, train_metrics: None):
        if hasattr(trainer.model, EXTEND_MODULE_NAME):
            extend_model = getattr(trainer.model, EXTEND_MODULE_NAME)
            delattr(trainer.model, EXTEND_MODULE_NAME)
            trainer.save_model(save_model_path)
            torch.save(extend_model.state_dict(), os.path.join(save_model_path, f'{EXTEND_MODULE_NAME}.bin'))
        else:
            trainer.save_model(save_model_path)

        # save all arguments as json file
        json_content = {
            'meta_arguments': Argument.field_name2value_jsonable(ArgumentPool.meta_arguments),
            'approach_arguments': Argument.field_name2value_jsonable(self.args),
            'data_arguments': Argument.field_name2value_jsonable(self.precessing_data.args),
            'training_arguments': Argument.field_name2value_jsonable(self.training_args),
        }

        with open(os.path.join(save_model_path, f'run_arguments.json'), mode='w') as f:
            json.dump(json_content, f, sort_keys=True, indent=4)

        if train_metrics is not None:
            with open(os.path.join(save_model_path, f'train_results.json'), mode='w') as f:
                json.dump(train_metrics, f, sort_keys=True, indent=4)

    def _process(self, dataset: Union[Tuple[Dataset],Dict[DatasetSplitType, Union[None,datasets.Dataset]]], *args, trainer: Trainer = None,
                 trainer_factory: type = None, is_save_model: bool = True, output_dir: str = None, **kwargs):
        if isinstance(dataset, tuple):
            dataset = self._request_datasets_to_dict(dataset=dataset)
        if dataset is None or len(dataset) == 0:
            return None

        if trainer_factory is None:
            trainer_factory = self._trainer_class

        if not isinstance(trainer_factory, Callable) and not issubclass(trainer_factory, Trainer):
            raise ValueError

        if output_dir is None:
            output_dir = self.training_args.save_model_path
        if not os.path.isdir(output_dir):
            os.system(f"mkdir -p {output_dir}")

        train_data = dataset[DatasetSplitType.TRAIN] if self.training_args.do_train else None
        dev_data = dataset[DatasetSplitType.VALIDATION] if self.training_args.do_eval else None
        if not isinstance(trainer, Trainer):
            trainer = trainer_factory(
                self.model,
                self.training_args,
                train_dataset=train_data,
                eval_dataset=dev_data,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics,
                callbacks=[DumpCallback(dump_path=output_dir), CudaCallback()]
            )

        # setattr(trainer, "current_approach", self)
        if self.training_args.do_train:
            train_result = trainer.train(**(kwargs.get('train', {})))
            train_metrics = train_result.metrics
            if is_save_model:
                print(f"Save trained model and its info to path '{output_dir}'")
                self._save_trained_model(trainer, output_dir, train_metrics)
        if self.training_args.do_eval:
            trainer.evaluate()

        if self.training_args.do_predict:
            test_dataset = dataset[DatasetSplitType.TEST]
            trainer.predict(test_dataset)

        # delattr(trainer, "current_approach")

    def _post_process(self, dataset: Dict[DatasetSplitType, Union[None, Dataset]], *args, **kwargs):
        if self.args.do_apply:
            self._default_application_launcher()
        if dataset is None or len(dataset) == 0:
            return None
        pass

    @abstractmethod
    def _prediction_to_output(self, prediction: np.ndarray, runtime: Optional[Dict[str, Any]] = None) -> Union[np.ndarray, List]:
        pass

    def _application(self, data: Data, runtime: Dict[str, Any]):
        dataset_size_limit = min(len(runtime["dataset"]) // 8, 10000)

        if "input" not in runtime["dataset"].column_names:
            raise ValueError

        dataset:Dataset = runtime["dataset"].map(lambda samples: self._tokenize(samples["input"], samples.get("label", None)), batched=True)
        trainer = self._trainer_class(args=self.training_args, model=self.model, tokenizer=self.tokenizer, compute_metrics=self._compute_metrics)
        loop_times = math.ceil(len(dataset) / dataset_size_limit)
        os.makedirs(runtime["output_dir"], exist_ok=True)
        tmp_file = os.path.join(runtime["output_dir"], "application_output_tmp.pkl")
        if os.path.isfile(tmp_file):
            os.system(f"rm {tmp_file}")
        for i in range(loop_times):
            print(f"**** Processing {i+1}th/{loop_times} Sub Dataset of {data.abbreviation} (size limit is {dataset_size_limit}) ****")
            if i != loop_times - 1:
                select_range = range(i*dataset_size_limit, (i+1)*dataset_size_limit)
            else:
                select_range = range(i*dataset_size_limit, len(dataset))
            result: PredictionOutput = trainer.predict(dataset.select(select_range))

            if os.path.isfile(tmp_file):
                with open(tmp_file, mode="rb") as f:
                    output_ = pickle.load(f)
                    output_.extend(self._prediction_to_output(result, runtime))
            else:
                output_ = self._prediction_to_output(result, runtime)

            with open(tmp_file, mode="wb") as f:
                pickle.dump(output_, f)

        with open(tmp_file, mode="rb") as f:
            output_ = pickle.load(f)

        print("Processing Finished!")
        # os.system(f"rm {tmp_file}")

        if "output" in runtime["dataset"].column_names:
            runtime["dataset"] = runtime["dataset"].remove_columns(column_names="output")
        runtime["dataset"] = runtime["dataset"].add_column(name='output', column=output_)
        for name in dataset.column_names:
            if name not in runtime["dataset"].column_names:
                runtime["dataset"] = runtime["dataset"].add_column(name=name, column=dataset[name])
        runtime["metrics"] = result.metrics
        return result

    def _compute_metrics(self, eval_predictions: EvalPrediction):
        tokenizer = self.tokenizer
        label_ids = eval_predictions.label_ids
        input_ids = eval_predictions.inputs
        assert input_ids is not None
        inputs = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
        inputs = np.array(tokenizer.batch_decode(inputs, skip_special_tokens=True))

        if self.precessing_data.task_type in (TaskType.CLASSIFICATION, TaskType.REGRESSIVE):
            # predictions = self._prediction_to_output(eval_predictions.predictions)
            # label_ids = self._prediction_to_output(label_ids)
            predictions = eval_predictions.predictions
            assert isinstance(predictions, np.ndarray) and isinstance(input_ids, np.ndarray)
            problem_type = self.plm.config.problem_type
            data_type = float
            if problem_type == "regression":
                predictions = predictions.flatten()

            elif problem_type == "single_label_classification":
                predictions = np.argmax(predictions, axis=-1)
                data_type = int

            elif problem_type == "multi_label_classification":
                predictions: np.ndarray = np.greater(predictions, 0).astype(int)
                data_type = int
            else:
                raise ValueError
            assert predictions.shape == label_ids.shape
            result = self.compute_metric(predictions.astype(data_type), label_ids.astype(data_type), inputs, self._prediction_to_output)

        elif self.precessing_data.task_type is TaskType.GENERATION:
            predictions: np.ndarray = eval_predictions.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            assert isinstance(predictions, np.ndarray)
            model: PreTrainedModel = self.model
            if predictions.shape[-1] == model.config.vocab_size:
                predictions = np.argmax(predictions, axis=-1)
            assert np.issubdtype(predictions.dtype, np.integer)
            # Add mean generated length
            result = {"gen_len": np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions])}

            predictions = self._prediction_to_output(predictions)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            labels = self._prediction_to_output(labels)

            if self.args.smoke_test:
                min_length = self.model.config.min_length
                if not isinstance(min_length, int):
                    min_length = 2
                min_length = max(2, min_length)
                for i in range(len(predictions)):
                    assert isinstance(predictions[i], str)
                    length = len(predictions[i].split(' '))
                    if length < min_length:
                        predictions[i] = " ".join([predictions[i], tokenizer.pad_token * (min_length - length)])

            result.update(self.compute_metric(predictions, labels, inputs))

        else:
            raise ValueError

        return result

    def metric_names(self, *args, data: Data = None, **kwargs):
        if data is None:
            data = self.precessing_data
        data_metric_names = data.metric_names
        return {
            'train': tuple([f'train_{n}' for n in data_metric_names["names"]]),
            'evaluation': tuple([f'eval_{n}' for n in data_metric_names["names"]])
        }

    def target_metric_for_tuning_hp(self, *args, data: Data = None, **kwargs):
        result = dict()
        if data is None:
            data = self.precessing_data
        target_metric = data.target_metric
        result['name'] = f'eval_{target_metric["name"]}'
        result['mode'] = "max" if target_metric["direction"] > 0 else "min"
        return result

    @property
    def tune_prepare(self) -> Dict[str, Any]:
        result = dict()
        result['num_samples'] = 4 if self.args.smoke_test else self.args.num_trials
        return result

    def tune_hyperparameter(self, dataset:Dict[DatasetSplitType, Dataset], *args, **kwargs):
        smoke_test = self.args.smoke_test
        tune_prepare = self.tune_prepare
        approach = self
        # approach.precessing_data = Name2DataClass["ppf"]()
        # approach.precessing_data._dataset = None
        # approach.precessing_data._metric = None // Can not instantionize the metric
        # update_instant_attribute(source=self, target=approach)
        trainer = self._trainer_class(
            # model=self.auto_classes["model"].from_pretrained(self._check_point()),
            model_init = lambda _: approach._model_init(),
            args = approach.training_args,
            train_dataset = dataset[DatasetSplitType.TRAIN],
            eval_dataset = dataset[DatasetSplitType.VALIDATION],
            data_collator = approach._data_collator_class(tokenizer=approach.tokenizer),
            tokenizer = approach.tokenizer,
            compute_metrics = approach._compute_metrics,
        )
        # trainer._memory_tracker = None
        self.release_model(trainer.model)
        torch.cuda.empty_cache()
        trail_name = approach.trail_name
        try:
            trainer.hyperparameter_search(
                hp_space=lambda _: tune_prepare["hp_space"],
                backend="ray",
                n_trials=tune_prepare["num_samples"],
                resources_per_trial={"cpu": 1, "gpu": approach.args.gpu_per_trial},
                scheduler=tune_prepare["scheduler"],
                keep_checkpoints_num=1,
                checkpoint_score_attr="training_iteration",
                stop = {"training_iteration": 1} if smoke_test else None,
                progress_reporter=tune_prepare["reporter"],
                name=f'tune_{trail_name}',
                log_to_file=False,
            )
        except Exception:
            breakpoint()
            # search(trainer)
            raise
# box = set()
# def search(obj):
#     from experiment.data.metrics.ppf_metric.ppf_metric import PPFMetric
#     if isinstance(obj, Data) or isinstance(obj, PPFMetric):
#         import traceback
#         print(traceback.format_exc())
#         print(obj)
#         return
#     if hasattr(obj, '__dict__'):
#         for k in obj.__dict__:
#             v = obj.__dict__[k]
#             if id(v) not in box:
#                 box.add(id(v))
#                 search(v)
#
#     return


class CudaCallback(TrainerCallback):
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # torch.cuda.empty_cache()
        # Thread(target=_clean_memory_cache, args=[args]).run()
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # torch.cuda.empty_cache()
        # Thread(target=_clean_memory_cache, args=[args]).run()
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # breakpoint()
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # torch.cuda.empty_cache()
        # Thread(target=_clean_memory_cache, args=[args]).run()
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Thread(target=_clean_memory_cache, args=[args]).run()
        return control


class DumpCallback(TrainerCallback):
    def __init__(self, dump_path:str=None, path_suffix_factory: Callable = None,*args, **kwargs):
        super().__init__( *args, **kwargs)
        self._dump_path = dump_path
        if isinstance(path_suffix_factory, Callable):
            self._path_suffix_factory = path_suffix_factory
        else:
            self._path_suffix_factory = lambda: ''

    @property
    def path_suffix_factory(self):
        return self._path_suffix_factory

    @path_suffix_factory.setter
    def path_suffix_factory(self, value: Callable):
        assert isinstance(value, Callable)
        self._path_suffix_factory = value

    @property
    def dump_path(self):
        # if isinstance(self._dump_path, str) and len(self._dump_path) > 0:
        #     return self._dump_path
        # if isinstance(self._dump_path_factory, Callable):
        #     self._dump_path = self._dump_path_factory()
        #     return self._dump_path
        return None

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # torch.cuda.empty_cache()
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.get("metrics", None)
        dump_path = self.dump_path if isinstance(self.dump_path, str) else args.save_model_path
        self._dump_metrics(metrics, "eval", dump_path, state)
        return control

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.get("metrics", None)
        dump_path = self.dump_path if isinstance(self.dump_path, str) else args.save_model_path
        self._dump_metrics(metrics, "test", dump_path, state)
        return control

    def _dump_metrics(self, metrics: Dict, metric_key_prefix: str, file_path: str, state: TrainerState):
        component = []
        # if isinstance(state.trial_params, dict) and "path_suffix_component" in state.trial_params:
        #     component = str(state.trial_params["path_suffix_component"])
        #     assert len(component) > 0
        #     component = [component,]
        path_suffix = self.path_suffix_factory()
        assert isinstance(path_suffix, str)
        if len(path_suffix) > 0:
            component.append(path_suffix)
        component = '_'.join(component)

        meta_metrics_key = f'{metric_key_prefix}_meta_metrics'
        if metrics is not None and meta_metrics_key in metrics:
            original_metrics = metrics
            metrics = OrderedDict(metrics)
            metrics.move_to_end(meta_metrics_key)
            metrics[meta_metrics_key] = metrics[meta_metrics_key].to_jsonable_dict()
            file_path = os.path.join(file_path, f'{component}_{metric_key_prefix}_results.json')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, mode='w') as f:
                json.dump(metrics, f, indent=4)
            original_metrics.pop(meta_metrics_key)



