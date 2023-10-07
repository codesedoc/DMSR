import os
from dataclasses import field
from enum import Enum
from typing import Optional, List, Any, Dict, Iterator, Callable, Set, Union, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from ..utils import tuning_hp_prepare_stpg
from src.nlps.data.data import MultiTaskDataset, merge_datasets, Data
from .dps_bart import  BartDPSConfig, BartDPSEncoder, BartDPSTokenizerFast, BartDPSModelForPTR
from .dps_t5 import T5DPSConfig, T5DPSEncoder, T5DPSTokenizerFast, T5DPSModelForPTR
from .dps_base import TaskCase, Encoder_Path_Type, DPSConfig, DPSEncoder, DPSTokenizer, DPSModelForPTR, DPSTokenizerFast

from src.nlps.approach import TransformerArgument, approach_register
from src.nlps.approach.transformer import DumpCallback, Transformer
from src.nlps.argument import argument_class, Argument
from src.nlps.data import DatasetSplitType, Name2DataClass, TextData
from src.nlps.utils.runtime import RunTime
from ..utils.ppf_utils import PGSTModelArgument


class DPSVariantType(Enum):
    ST2PG = "st2pg"
    PG2ST = "pg2st"
    WO_SENSE = "wo_sense"
    WO_SENTI = "wo_senti"
    INCORPORATE = "incorporate"
    BASELINE = "baseline"

    def __str__(self):
        return self.value


Name2BackboneClass = {
    "bart": {
        "config": BartDPSConfig,
        "tokenizer": BartDPSTokenizerFast,
        "model": BartDPSModelForPTR
    },
    "t5": {
        "config": T5DPSConfig,
        "tokenizer": T5DPSTokenizerFast,
        "model": T5DPSModelForPTR
    }
}


@argument_class
class DPSModelArgument(PGSTModelArgument):
    pass


@argument_class
class DPSArgument(TransformerArgument):
    variant: str = field(
        default="self",
        metadata={
            'help': "Specify one of the three experiment settings",
            'choices': str([e.value for e in DPSVariantType])
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        from ..data import MscocoStyleTransfer, YelpStyleTransfer
        _generation_to_output = lambda pred, **kwargs: kwargs.pop("tokenizer").batch_decode(pred, **kwargs)
        runtime = RunTime()
        self.name2auxiliary_task: Dict[str, TaskCase] = {
            MscocoStyleTransfer.abbreviation: TaskCase(name=MscocoStyleTransfer.abbreviation,
                                                       data=None,
                                                       prediction_to_output=_generation_to_output,
                                                       path_type=Encoder_Path_Type.PG),

            YelpStyleTransfer.abbreviation: TaskCase(name=YelpStyleTransfer.abbreviation, data=None,
                                                     prediction_to_output=_generation_to_output,
                                                     path_type=Encoder_Path_Type.ST),
        }

        self.variant: DPSVariantType = DPSVariantType(self.variant)

        self.name2taskcase: Dict[str, TaskCase] = {
            runtime.data.abbreviation: TaskCase(name=runtime.data.abbreviation, data=runtime.data,
                                        prediction_to_output=_generation_to_output,
                                        path_type=None,
                                        is_main_task=True),
        }

        if self.variant == DPSVariantType.WO_SENSE:
            self.name2taskcase[runtime.data.abbreviation].path_type = Encoder_Path_Type.ST

        elif self.variant == DPSVariantType.WO_SENTI:
            self.name2taskcase[runtime.data.abbreviation].path_type = Encoder_Path_Type.PG

        elif self.variant in {DPSVariantType.ST2PG, DPSVariantType.PG2ST}:
            self.name2taskcase[runtime.data.abbreviation].path_type = Encoder_Path_Type(self.variant.value)

        elif self.variant == DPSVariantType.BASELINE:
            self.name2taskcase[runtime.data.abbreviation].path_type = Encoder_Path_Type.BASE

        elif self.variant == DPSVariantType.INCORPORATE:
            # self.name2taskcase.pop("ppf")
            raise
        else:
            raise ValueError

        self.name2all_task = {}
        self.name2all_task.update(self.name2taskcase)
        self.name2all_task.update(self.name2auxiliary_task)


class MultiTaskSampler(Sampler[int]):
    def __init__(self, sampler: Sampler[int], batch_size: int, dataset: Dataset,
                 task_names: Set[str]) -> None:
        super().__init__(dataset)
        self._sampler = sampler
        self._batch_size = batch_size
        self._task_names = task_names
        self._dataset = dataset

    def __iter__(self) -> Iterator[List[int]]:
        final_index_order = []
        sub_sampler: Dict[str, list] = {n: [] for n in self._task_names}
        data_source: Dataset = self._dataset
        for i in iter(self._sampler):
            task_name = data_source[i]['task']
            sub_sampler[task_name].append(i)
            if len(sub_sampler[task_name]) == self._batch_size:
                final_index_order.extend(sub_sampler[task_name])
                sub_sampler[task_name] = []
        for i in final_index_order:
            yield i

    def __len__(self) -> int:
        return len(self._dataset)


class DPSDataCollator(DataCollatorForSeq2Seq):
    # def __init__(self, name2taskcase: Dict[str, TaskCase]):
    #     super().__init__()
    #     self.name2taskcase = name2taskcase

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        t0 = features[0]['task']
        for f in features:
            try:
                assert f['task'] == t0
            except Exception:
                print(f['task'])
                print(t0)
                print([fe["task"] for fe in features])
                raise

        stringed_column = set()
        for k, v in features[0].items():
            if isinstance(v, str) and v.startswith("Stringed_Column:"):
                stringed_column.add(k)

        for f in features:
            for c_m in stringed_column:
                v = f[c_m]
                stringed_value = v.split("Stringed_Column:")[1]
                f[c_m] = eval(stringed_value)

            if "task" in f.keys():
                f.pop("task")
            if "label" in f.keys():
                f.pop("label")

        # if "decoder_attention_mask" in features[0]:
        #     for f in features:
        #         f.pop("decoder_attention_mask")
        #     batch = super().__call__(features, return_tensors)
        #     labels = batch["labels"]
        #     attention_mask = batch["attention_mask"]
        #     batch["decoder_attention_mask"] = torch.where(labels!=-100, torch.ones(labels.size()), torch.zeros(labels.size())).to(dtype=attention_mask.dtype, device=attention_mask.device)
        # else:
        #     batch = super().__call__(features, return_tensors)
        batch = super().__call__(features, return_tensors)
        batch["task"] = t0
        return batch


class DPSTrainer(Seq2SeqTrainer):
    _evaluate_end_call_back: Callable = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, DumpCallback):
                config: DPSConfig = self.model.config
                callback.path_suffix_factory = lambda: config.dynamic_taskcase.name

    @classmethod
    @property
    def evaluate_end_call_back(cls) -> Callable:
        return cls._evaluate_end_call_back

    @classmethod
    def set_evaluate_end_call_back(cls, value: Callable):
        assert isinstance(value, Callable)
        cls._evaluate_end_call_back = value

    def _get_multitask_sampler(self, base_sampler, batch_size, dataset):
        assert isinstance(base_sampler, Sampler)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(dataset, Dataset)
        config: DPSConfig = self.model.config
        sampler = MultiTaskSampler(sampler=base_sampler, batch_size=batch_size, dataset=dataset,
                                   task_names=config.task_names)
        return sampler

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[Sampler]:
        base_sampler = super()._get_eval_sampler(eval_dataset)
        sampler = self._get_multitask_sampler(base_sampler, self.args.eval_batch_size, eval_dataset)
        return sampler

    def _get_train_sampler(self) -> Optional[Sampler]:
        base_sampler = super()._get_train_sampler()
        sampler = self._get_multitask_sampler (base_sampler, self.args.train_batch_size, self.train_dataset)
        return sampler

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs
    ) -> Dict[str, Any]:
        # breakpoint()
        result = None
        config: DPSConfig = self.model.config
        for task, case in config.name2taskcase.items():
            config.dynamic_taskcase = case
            print(f"**** Evaluating for Task: {task} ****")
            output = super().evaluate(eval_dataset=case.data.dataset(DatasetSplitType.VALIDATION), **gen_kwargs)
            if result is None:
                result = dict()
            result[case.name] = output
        call_back = self.evaluate_end_call_back
        assert isinstance(call_back, Callable)
        result = call_back(result)
        assert result is not None
        return result


@approach_register
class DisentangledPS(Transformer):
    _abbreviation = 'dps'
    _training_arg_class = Seq2SeqTrainingArguments
    _argument_class = DPSArgument
    _trainer_class = DPSTrainer
    _data_collator_class = DPSDataCollator
    _model_argument_class = DPSModelArgument

    _auxiliary_name2taskcase = {

    }
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    @property
    def check_point(self):
        checkpoint = super().check_point
        if not os.path.isdir(checkpoint):
            checkpoint = self.model_args.backbone
        return checkpoint

    def _config_init(self, *args, **kwargs):
        config = super()._config_init(*args, **kwargs)
        args: DPSArgument = self.args
        config.max_length = max([c.data.max_length for c in args.name2taskcase.values()])
        config.min_length = min([c.data.min_length for c in args.name2taskcase.values()])
        return config

    def _first_stage(self, transformer: DPSModelForPTR) -> DPSModelForPTR:
        args: DPSArgument = self.args
        config: DPSConfig = transformer.config
        name2taskcase = config.name2taskcase
        dynamic_taskcase = config.dynamic_taskcase
        config.name2taskcase = args.name2auxiliary_task
        config.dynamic_taskcase = None

        DPSTrainer.set_evaluate_end_call_back(lambda x: x)
        dataset = self._multi_task_dataset(args.name2auxiliary_task)
        output_dir = os.path.join(self.training_args.save_model_path, 'auxiliary_task')
        self._process(dataset=dataset, is_save_model=False, output_dir=output_dir)

        config.name2taskcase = name2taskcase
        config.dynamic_taskcase = dynamic_taskcase

        print('*********** Finish First Stage ***********')

        return transformer

    def _model_init(self, *args, **kwargs):
        transformer: DPSModelForPTR = super()._model_init(*args, **kwargs)
        args: DPSArgument = self.args
        if args.variant is not DPSVariantType.BASELINE:
            transformer = self._first_stage(transformer)
        config: DPSConfig = transformer.config
        config.name2taskcase = args.name2taskcase
        config.dynamic_taskcase = None
        DPSTrainer.set_evaluate_end_call_back(lambda output: output[self.precessing_data.abbreviation])
        return transformer

    def _prediction_to_output(self, prediction: np.ndarray, runtime: Optional[Dict[str, Any]] = None) -> Union[
        np.ndarray, List]:
        case: TaskCase = self.model.config.dynamic_taskcase
        kw = {"skip_special_tokens": True}
        if isinstance(case, TaskCase):
            decoder = case.prediction_to_output
            kw["tokenizer"] = self.tokenizer
        else:
            decoder = self.tokenizer.batch_decode
        result = decoder(prediction, **kw)
        return result

    def release_model(self, model=None):
        super().release_model(model)

    @property
    def auto_classes(self):
        model_args: PGSTModelArgument = self.model_args
        return Name2BackboneClass[model_args.backbone_name]

    def pre_tokenization_call_back(self, sequence):
        tokenizer: DPSTokenizerFast = self.tokenizer
        return tokenizer.preprocess_input_sequence(sequence)

    def _preprocess(self, samples: Union[Dataset, Dict[str, Any]], *args, **kwargs):
        from src.nlps.data import TextData
        data: TextData = self.precessing_data
        kwargs["max_length"] = data.max_length
        result = super()._preprocess(samples, *args, **kwargs)
        result["task"] = [self.precessing_data.abbreviation] * len(result["input_ids"])
        return result

    def _multi_task_dataset(self, taskcase_dict: Dict[str, TaskCase]):
        dataset_dicts = []
        for name, case in taskcase_dict.items():
            if self.precessing_data.abbreviation == name:
                case.data = self.precessing_data
            if case.data is None:
                case.data = Name2DataClass[case.name]()
            assert isinstance(case.data, TextData)
            dd = super()._request_datasets_to_dict(data=case.data)
            for s, d in dd.items():
                dd[s] = d.rename_column("task", MultiTaskDataset.task_column_name)
            dataset_dicts.append(dd)

        multi_task_dataset = merge_datasets(tuple(dataset_dicts))

        assert isinstance(multi_task_dataset, dict) \
               and len(multi_task_dataset.items()) > 0 \
               and isinstance(list(multi_task_dataset.values())[0], Dataset)

        for s, d in multi_task_dataset.items():
            multi_task_dataset[s] = d.rename_column(MultiTaskDataset.task_column_name, "task")

        return multi_task_dataset

    def _request_datasets_to_dict(self, data: Data = None, dataset: Tuple[Dataset] = None, **kwargs) -> Optional[Dict[DatasetSplitType, Dataset]]:
        args: DPSArgument = self.args
        multi_task_dataset = self._multi_task_dataset(args.name2taskcase)
        return multi_task_dataset

    def _process(self, *args, **kwargs):
        super()._process(*args, **kwargs)

    @classmethod
    def collect_argument(cls):
        super().collect_argument()
        Argument.update_defaults_for_fields(cls.training_arg_class, {
            # 'evaluation_strategy': "epoch",
            'save_total_limit': 3,
            'weight_decay': 0.01,
            'predict_with_generate': True,
            'use_mps_device': False
        })

    @property
    def tune_prepare(self):
        result = super().tune_prepare
        result = tuning_hp_prepare_stpg(self, result)
        return result

    @property
    def trail_name(self):
        args: DPSArgument = self.args
        return f'{self.abbreviation}_{args.dataset}_{args.variant.value}'

    @property
    def path_component(self):
        args: DPSArgument = self.args
        return f'{super().path_component}/{self.model_args.backbone}/{args.variant.value}'
