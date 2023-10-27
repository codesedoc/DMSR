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
from nlpx.data import TextData


@data_register
class PositivePsychologyFrames(TextData):

    _abbreviation = 'ppf'
    _metric_name_path = 'experiment/data/metrics/ppf_metric'
    _task_type = TaskType.GENERATION

    def _preprocess(self,  splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT, *args, **kwargs):
        name2splits: Dict[str, DatasetSplitType] = {
            'wholetrain': DatasetSplitType.TRAIN,
            'wholetest': DatasetSplitType.TEST,
            'wholedev': DatasetSplitType.VALIDATION
        }
        extensions = 'csv'
        for n, s in name2splits.items():
            raw_path = os.path.join(self.raw_dir, f'{n}.{extensions}')
            if not os.path.isfile(raw_path):
                raise ValueError

            output_path = self._preprocessed_files[s]
            output_dir = os.path.split(output_path)[0]
            os.makedirs(output_dir, exist_ok=True)
            os.system(f"cp -a {raw_path} {output_path}")
            # df = pd.read_csv(raw_path, index_col=0)
            # df.to_csv(output_path, index=False)

    @property
    def an_sample(self) -> Tuple[str]:
        return "This is a example", "This is a example"

    @property
    def input_column_name(self):
        return "original_text"

    @property
    def label_column_name(self):
        return "reframed_text"

    def extract_input_label_from_samples(self, samples: Union[Dataset, Dict[str, Any]], *args, **kwargs):
        if isinstance(samples, Dataset):
            names = samples.column_names
        elif isinstance(samples, MutableMapping):
            names = samples.keys()
        else:
            raise ValueError
        if "reframed_text" in names:
            return samples["original_text"], samples["reframed_text"]
        else:
            return samples["original_text"],

    def _dataset_for_application(self, runtime: Dict[str, Any], *args, **kwargs) -> Dataset:
        dataset = self.dataset(DatasetSplitType.VALIDATION)[0]
        column_names = dataset.column_names
        remove_column_names = [c for c in column_names if c != self.input_column_name]
        if len(remove_column_names) > 0:
            dataset.remove_columns(remove_column_names)
        dataset = dataset.rename_column(original_column_name="original_text", new_column_name="input")
        return dataset

    @property
    def target_metric(self):
        return {
            "name": "perplexity",
            "direction": -1,
        }
            


