import copy
import math
import pickle
import random
from dataclasses import field

import datasets
from datasets import Dataset

from nlpx.argument import DataArgument, argument_class
from nlpx.data import Data, data_register, DatasetSplitType, TaskType, GeneralDataset, DataContainer, \
    DataDirCategory, TextData
from typing import Dict, Tuple, Union, Any, MutableMapping
import os
import pandas as pd

from nlpx.data.data import ALL_DATASET_SPLIT
from nlpx.utils.utils import max_length_of_sequences


class Pseudo(TextData):
    _abbreviation = 'pseudo'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._max_length = None


class SingleSourceStyleTransfer(Pseudo):
    @property
    def raw_data_file_name(self):
        raise ValueError

    def _preprocess(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT, *args, **kwargs):
        splits2names: Dict[DatasetSplitType, str] = {
            DatasetSplitType.TRAIN: 0.8,
            DatasetSplitType.TEST: 0.1,
            DatasetSplitType.VALIDATION: 0.1
        }
        file_name = self.raw_data_file_name
        raw_path = os.path.join(self.raw_dir, f'{file_name}')
        if not os.path.isfile(raw_path):
            raise ValueError(f"{raw_path} is not exist!")
        dataset = datasets.load_dataset("csv", data_files=raw_path)["train"]
        data_size = len(dataset)
        rate_start_point = 0
        for s, rate in splits2names.items():
            if s not in self._preprocessed_files:
                continue
            output_path = self._preprocessed_files[s]
            output_dir = os.path.split(output_path)[0]
            os.makedirs(output_dir, exist_ok=True)

            rate_end_point = rate_start_point + rate
            split_dataset = Dataset.from_dict(
                dataset[math.floor(rate_start_point * data_size): math.floor(rate_end_point * data_size)])
            split_dataset.to_csv(output_path, index=False)
            rate_start_point += rate

    @property
    def an_sample(self) -> Tuple[str]:
        return "i was sadly mistaken .", "i was sadly mistaken ."

    @property
    def input_column_name(self):
        return "input"

    @property
    def label_column_name(self):
        return "reference"


@data_register
class MscocoStyleTransfer(SingleSourceStyleTransfer):
    _abbreviation = os.path.join(Pseudo._abbreviation, 'mscoco_style_transfer')
    _metric_name_path = 'experiment/data/metrics/ppf_metric'
    _task_type = TaskType.GENERATION

    @property
    def raw_data_file_name(self):
        return 'sentiment_bart-base_mscoco_train.csv'


@data_register
class Parabk2StyleTransfer(SingleSourceStyleTransfer):
    _abbreviation = os.path.join(Pseudo._abbreviation, 'parabk2_style_transfer')
    _metric_name_path = 'experiment/data/metrics/ppf_metric'
    _task_type = TaskType.GENERATION

    @property
    def raw_data_file_name(self):
        return 'sentiment_bart-base_parabk2_train.csv'


@data_register
class WikiansStyleTransfer(SingleSourceStyleTransfer):
    _abbreviation = os.path.join(Pseudo._abbreviation, 'wikians_style_transfer')
    _metric_name_path = 'experiment/data/metrics/ppf_metric'
    _task_type = TaskType.GENERATION

    @property
    def raw_data_file_name(self):
        return 'sentiment_bart-base_wikians_train.csv'


@data_register
class YelpStyleTransfer(SingleSourceStyleTransfer):
    _abbreviation = os.path.join(Pseudo._abbreviation, 'yelp_style_transfer')
    _metric_name_path = 'experiment/data/metrics/ppf_metric'
    _task_type = TaskType.GENERATION

    @property
    def raw_data_file_name(self):
        return 'glue-stsb_bart-base_yelp_senti_train_rate-0.05_154.csv'

    def _preprocess(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT, *args, **kwargs):
        splits2names: Dict[DatasetSplitType, str] = {
            DatasetSplitType.TRAIN: 0.8,
            DatasetSplitType.TEST: 0.1,
            DatasetSplitType.VALIDATION: 0.1
        }
        file_name = self.raw_data_file_name
        raw_path = os.path.join(self.raw_dir, f'{file_name}')
        if not os.path.isfile(raw_path):
            raise ValueError
        dataset = datasets.load_dataset("csv", data_files=raw_path)["train"]
        data_size = len(dataset)
        rate_start_point = 0
        for s, rate in splits2names.items():
            if s not in self._preprocessed_files:
                continue
            output_path = self._preprocessed_files[s]
            output_dir = os.path.split(output_path)[0]
            os.makedirs(output_dir, exist_ok=True)

            rate_end_point = rate_start_point + rate
            split_dataset = Dataset.from_dict(
                dataset[math.floor(rate_start_point * data_size): math.floor(rate_end_point * data_size)])
            if s != DatasetSplitType.TRAIN:
                unique_input = set()
                final_samples = []
                for sample in split_dataset:
                    if sample["input"] in unique_input:
                        continue
                    unique_input.add(sample["input"])
                    final_samples.append(sample)
                if len(final_samples) > 0:
                    Dataset.from_list(final_samples).to_csv(output_path, index=False)
            else:
                split_dataset.to_csv(output_path, index=False)
            rate_start_point += rate




