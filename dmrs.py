from collections import OrderedDict
import json
import os
from dataclasses import field, dataclass, asdict
from enum import Enum, StrEnum, auto
from pathlib import Path
import traceback
from typing import Optional, List, Any, Dict, Iterator, Callable, Set, Union, Tuple
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import RandomSampler, Sampler
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, TrainingArguments, TrainerState, TrainerControl, TrainerCallback, EvalPrediction, CONFIG_MAPPING, AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate.data_loader import DataLoaderShard
from nlpe import Approach, ArgumentPool, ArgumentFactory, Pool, Data, TextData, DatasetSplitCategory
from nlpe.utils import Glossary, global_logger
from dataset import DatasetGlossaryId2VariantGlossary, GlossaryIDColumnName, InputColumnName, LabelColumnName
from model import DMSRModel, BackboneArgument, VariantGlossaryEnum, dynamic_layers_in_forward, BackboneName
from utils import log_segment


class DMSR(Approach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        backbone_arg: BackboneArgument = ArgumentPool()["backbone_argument"]
        checkpoint = backbone_arg.checkpoint
        self.logger = global_logger()
        try: 
            model_type_str = backbone_arg.model_type_str
            config = CONFIG_MAPPING[model_type_str].from_pretrained(checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, config=config)
            self.model: DMSRModel = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, config=config)
            self.logger.info(f"********** Successfully Load Model From {checkpoint} **********")
            log_segment(self.logger.debug, "Config", str(self.model.config))
            log_segment(self.logger.debug, "Tokenizer", str(self.tokenizer))
            log_segment(self.logger.debug, "Model", str(self.model))
        except:
            self.tokenizer = None
            self.model: DMSRModel = None
            raise ValueError(f"Can not load from the checkpoint '{checkpoint}'")

    def _init_trainer(self, data: TextData):
        trainer_arg: TrainingArguments = ArgumentPool()["trainer_argument"]
        return DMSRTrainer(
                        model=self.model,
                        args=trainer_arg,
                        data_collator=DataCollatorForSeq2Seq(self.tokenizer),
                        train_dataset=self.tokenization(data, DatasetSplitCategory.TRAIN),
                        eval_dataset=self.tokenization(data, DatasetSplitCategory.VALIDATION),
                        tokenizer=self.tokenizer,
                        compute_metrics=self._compute_metrics,
                        callbacks=[DumpCallback()]
        )
    
    def tokenization(self, data: TextData, split: DatasetSplitCategory) -> Dataset:
        backbone_arg: BackboneArgument = ArgumentPool()["backbone_argument"]
        def _tokenize(samples: Dict):
            samples = dict(samples)
            assert isinstance(next(iter(samples.values())), list)
            self.logger.info(f"Tokeinze the Samples in Split '{split}' of Dataset '{data.dataset_name}'")
            input_column = samples[InputColumnName]
            if backbone_arg.backbone == BackboneName.T5:
                input_column = ["rewrite positively: " + i for i in input_column]
            inputs = self.tokenizer(input_column)
            if LabelColumnName in samples:
                inputs["labels"] = self.tokenizer(samples[LabelColumnName])["input_ids"]
            for i, input_text in enumerate(input_column[:min(5, len(samples))]):
                self.logger.debug(f"Input {i}: {input_text}")
                self.logger.debug(f"Label {i}: {samples[LabelColumnName][i]}")
                for name, value in inputs.items():
                    self.logger.debug(f"{name} {i}: {value[i]}")
            return inputs
        return data.load_dataset(split).map(_tokenize, batched=True)
    
    def verify_model_state(self) -> bool:
        result = True
        if self.model.backbone_arg.variant == VariantGlossaryEnum.BASE.value:
            return self.model.pg_layers == None and self.model.st_layers == None
        
        for pg_p, st_p, d_p in zip(self.model.pg_layers.parameters(True), self.model.st_layers.parameters(True), self.model.dynamic_layers.parameters(True)):
            if torch.all(pg_p == st_p) or torch.all(pg_p == d_p) or torch.all(st_p == d_p):
                trainer_arg: TrainingArguments = ArgumentPool()["trainer_argument"]
                self.logger.warning(f"The pg_layers, st_layers, and dynamic_layers have same parameters!")
                result = not trainer_arg.do_train
                break
        return result
               
    def _process(self, data: TextData, *args, stage=1, **kwargs):
        trainer_arg: Seq2SeqTrainingArguments = ArgumentPool()["trainer_argument"]
        
        data.statistic_all_texts(tokenizor=lambda text: self.tokenizer(text)["input_ids"])
        self.logger.info(f"Max lengtg of {data.dataset_name} is: {data.max_length}")
        self.logger.info(f"Min lengtg of {data.dataset_name} is: {data.min_length}")
        trainer_arg.generation_max_length = data.max_length + data.min_length
        trainer: Seq2SeqTrainer = self._init_trainer(data)
        match stage:
            case 1:
                if self.model.backbone_arg.variant == VariantGlossaryEnum.BASE.value:
                    self.logger.info(f"Variant ({VariantGlossaryEnum.BASE.value}) will skil stage 1")
                else:
                    if trainer_arg.do_train:
                        self.logger.info(f"********** Training {data.dataset_name} **********")
                        trainer.train()
            case 2:
                test_dataset = self.tokenization(data, DatasetSplitCategory.TEST)
                assert self.verify_model_state()
                if trainer_arg.do_train:
                    self.logger.info(f"********** Training {data.dataset_name} **********")
                    trainer.train()
                    trainer.save_model(Path(trainer_arg.output_dir))
                if trainer_arg.do_eval:
                    self.logger.info(f"********** Evaluating {data.dataset_name} **********")
                    trainer.evaluate()
                if trainer_arg.do_predict:
                    self.logger.info(f"********** Testing {data.dataset_name} **********")
                    trainer.predict(test_dataset=test_dataset)
            case _:
                raise ValueError()
            
    def _compute_metrics(self, eval_predictions: EvalPrediction):
        meta_arg = ArgumentPool()["meta_argument"]
        tokenizer = self.tokenizer
        label_ids = eval_predictions.label_ids
        input_ids = eval_predictions.inputs
        assert input_ids is not None
        # Replace -100 in the labels as we can't decode them.
        map_ids2sequences = lambda ids: tokenizer.batch_decode(np.where(ids != -100, ids, tokenizer.pad_token_id), skip_special_tokens=True)
        inputs = map_ids2sequences(input_ids)
        predictions: np.ndarray = eval_predictions.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        assert isinstance(predictions, np.ndarray)
        model: DMSRModel = self.model
        if predictions.shape[-1] == model.config.vocab_size:
            predictions = np.argmax(predictions, axis=-1)
        assert np.issubdtype(predictions.dtype, np.integer)
        result = OrderedDict()
        result["avg_gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions])
        # self.logger.info(str(predictions))
        predictions = map_ids2sequences(predictions)
        labels = map_ids2sequences(label_ids)
        
        assert len(inputs) == len(labels) == len(predictions)
        items = []
        for i, l, p in zip(inputs, labels, predictions):
            if len(p.strip()) == 0:
                p = 'null null null'
            items.append(dict(input=i, reference=l, prediction=p))
        # file_path = Path(ArgumentPool()["trainer_argument"].output_dir, "eval_generations.json")
        # file_path.parent.mkdir(exist_ok=True)
        # file_path.write_text(json.dumps(items, indent=4))
        # if meta_arg.debug:
        self.logger.info(json.dumps(items[:min(len(items), 10)], indent=4))
        
        self.logger.info("********** Evaluate Predictions **********")
        result["prediction"] = (self.processing_data.evaluate(predictions=predictions, references=labels, inputs=inputs))
        self.logger.info("********** Evaluate Inputs **********")
        result["input"] = (self.processing_data.evaluate(predictions=inputs, references=labels, inputs=inputs))
        self.logger.info("********** Evaluate References **********")
        result["reference"] = (self.processing_data.evaluate(predictions=labels, references=labels, inputs=inputs))
        
        # if meta_arg.debug:
        self.logger.info(json.dumps(result, indent=4))
            
        result["examples"] = items
        return result
     

class MergedDatasetSampler(RandomSampler):
    def __init__(self, sampler: Sampler[int], batch_size: int, dataset: Dataset) -> None:
        super().__init__(dataset)
        self._sampler = sampler
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        final_index_order = []
        sub_sampler: Dict[str, list] = dict()
        data_source: Dataset = self.data_source
        if GlossaryIDColumnName in data_source[0]:
            for i in iter(self._sampler):
                glossary = data_source[i][GlossaryIDColumnName]
                if glossary not in sub_sampler:
                    sub_sampler[glossary] = []
                sub_sampler[glossary].append(i)
                if len(sub_sampler[glossary]) == self._batch_size:
                    final_index_order.extend(sub_sampler[glossary])
                    sub_sampler[glossary] = []
        else:
            final_index_order = list(iter(self._sampler))
        for i in final_index_order:
            yield i


class DMSRTrainer(Seq2SeqTrainer):
    def _get_multitask_sampler(self, base_sampler, batch_size, dataset):
        assert isinstance(base_sampler, Sampler)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(dataset, Dataset)
        sampler = MergedDatasetSampler(sampler=base_sampler, batch_size=batch_size, dataset=dataset)
        return sampler


    def get_train_dataloader(self) -> DataLoaderShard:
        dataloder: DataLoaderShard = super().get_train_dataloader()
        assert isinstance(dataloder, DataLoaderShard)
        dataloder.set_sampler(self._get_multitask_sampler(dataloder.get_sampler(), self.args.train_batch_size, dataloder.base_dataloader.dataset))
        return dataloder
    
    def get_eval__dataloader(self) -> DataLoaderShard:
        dataloder: DataLoaderShard = super().get_eval__dataloader()
        assert isinstance(dataloder, DataLoaderShard)
        dataloder.set_sampler(self._get_multitask_sampler(dataloder.get_sampler(), self.args.eval_batch_size, dataloder.base_dataloader.dataset)) 
        return dataloder
    
    def get_test__dataloader(self) -> DataLoaderShard:
        dataloder: DataLoaderShard = super().get_eval__dataloader()
        assert isinstance(dataloder, DataLoaderShard)
        dataloder.set_sampler(self._get_multitask_sampler(dataloder.get_sampler(), self.args.eval_batch_size, dataloder.base_dataloader.dataset)) 
        return dataloder
    
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns.append(GlossaryIDColumnName)
    
    def training_step(self, model: DMSRModel, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if GlossaryIDColumnName in inputs:
            glossary_id = set(inputs.pop(GlossaryIDColumnName).tolist())
            assert len(glossary_id) == 1
            glossary_id = glossary_id.pop()
            variant_glossary = DatasetGlossaryId2VariantGlossary[glossary_id]
        else:
            backbone_arg: BackboneArgument = ArgumentPool()["backbone_argument"]
            variant_glossary = backbone_arg.variant
        tmp_layers = model.dynamic_layers
        model.set_dynamic_layers(dynamic_layers_in_forward(model, variant_glossary=variant_glossary))
        result = super().training_step(model, inputs)
        model.set_dynamic_layers(tmp_layers)
        return result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None, **gen_kwargs):
        tmp_layers = model.dynamic_layers
        model.set_dynamic_layers(dynamic_layers_in_forward(model, variant_glossary=ArgumentPool()["backbone_argument"].variant))
        result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        model.set_dynamic_layers(tmp_layers)
        return result
    
    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     assert isinstance(self.model, DMSRModel)
    #     self.model.backbone_encoder.backbone_layers = None
    #     super().save_model(output_dir, _internal_call)
        

class DumpCallback(TrainerCallback):
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # torch.cuda.empty_cache()
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # breakpoint()
        metrics = kwargs.get("metrics", None)
        dump_path = Path(args.output_dir, "eval_result.json")
        self._dump_metrics(metrics, dump_path)
        return control

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.get("metrics", None)
        dump_path = Path(args.output_dir, "test_result.json")
        self._dump_metrics(metrics, dump_path)
        return control

    def _dump_metrics(self, metrics: Dict, file_path: Path):
        if metrics is not None:
            file_path.parent.mkdir(exist_ok=True)
            file_path.write_text(json.dumps(metrics, indent=4))