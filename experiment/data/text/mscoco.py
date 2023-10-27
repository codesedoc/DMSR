import copy
import pickle
import random

import datasets
from datasets import Dataset

from nlpx.data import Data, data_register, DatasetSplitType, TaskType, GeneralDataset, DataContainer, DataDirCategory
from typing import Dict, Tuple, Union, Any, MutableMapping
import os
import pandas as pd

from nlpx.data.data import ALL_DATASET_SPLIT
from nlpx.utils.utils import max_length_of_sequences


@data_register
class Mscoco(Data):
    _abbreviation = 'mscoco'
    _metric_name_path = 'bleu'
    _task_type = TaskType.GENERATION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_length = None

    def _preprocess(self):
        name2splits: Dict[str, DatasetSplitType] = {
            'train': DatasetSplitType.TRAIN,
            'test': DatasetSplitType.TEST,
            'validation': DatasetSplitType.VALIDATION
        }
        extensions = 'csv'
        for n, s in name2splits.items():
            raw_path = os.path.join(self.raw_dir, f'{n}.{extensions}')
            if not os.path.isfile(raw_path):
                raise ValueError

            output_path = self._preprocessed_files[s]
            output_dir = os.path.split(output_path)[0]
            os.makedirs(output_dir, exist_ok=True)

            assert os.system(f"cp -a {raw_path} {output_dir}") == 0

    @property
    def an_sample(self) -> Tuple[str]:
        return "This is a example", "This is a example"

    @property
    def input_column_name(self):
        return "source"

    @property
    def label_column_name(self):
        return "target"

    def extract_input_label_from_samples(self, samples: Union[Dataset, Dict[str, Any]], *args, **kwargs):
        if isinstance(samples, Dataset):
            names = samples.column_names
        elif isinstance(samples, MutableMapping):
            names = samples.keys()
        else:
            raise ValueError
        if "target" in names:
            return samples["source"], samples["target"]
        else:
            return samples["source"],

    def _load_dataset(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT) -> Dict[DatasetSplitType, Dataset]:
        result = super()._load_dataset(splits)
        self._max_length=max_length_of_sequences(list(result.values()))
        return result

    def _dataset_for_application(self, runtime: Dict[str, Any], *args, **kwargs) -> Dataset:
        dataset = self.dataset(DatasetSplitType.VALIDATION)[0]
        column_names = dataset.column_names
        remove_column_names = [c for c in column_names if c != self.input_column_name]
        if len(remove_column_names) > 0:
            dataset.remove_columns(remove_column_names)
        dataset = dataset.rename_column(original_column_name="source", new_column_name="input")
        return dataset

    @property
    def max_length(self):
        if not isinstance(self._max_length, int):
            self._max_length = max_length_of_sequences(self, dataset=list(self._dataset.values()))
        return self._max_length

    def _compute_metrics(self, predictions, labels, *args, **kwargs):
        predictions = [p.split() for p in predictions]
        references = [[l.split()] for l in labels]
        _metric = self._load_metric()
        result = _metric.compute(predictions=predictions, references=references)
        return {"bleu": result["bleu"]}

            


