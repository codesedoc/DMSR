import copy
import pickle
import random
from dataclasses import field

import datasets
from datasets import Dataset

from nlpx.argument import DataArgument, argument_class
from nlpx.data import Data, data_register, DatasetSplitType, TaskType, GeneralDataset, DataContainer, DataDirCategory
from typing import Dict, Tuple, Union, Any, MutableMapping
import os
import pandas as pd

from nlpx.data.data import ALL_DATASET_SPLIT
from nlpx.utils.utils import max_length_of_sequences


@argument_class
class YelpSentimentArgument(DataArgument):
    sampling_rate: float = field(
        default=-1,
        metadata={'help': "Specify the sampling rate of original dataset for generate pseudo style transfer dataset."}
    )

@data_register
class YelpSentiment(Data):
    _abbreviation = 'yelp_senti'
    _metric_name_path = 'accuracy'
    _task_type = TaskType.CLASSIFICATION
    _argument_class = YelpSentimentArgument

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_length = None

    def _preprocess(self,  splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT, *args, **kwargs):
        splits2names: Dict[str, DatasetSplitType] = {
            DatasetSplitType.TRAIN: 'sentiment.train',
            DatasetSplitType.TEST: 'sentiment.test',
            DatasetSplitType.VALIDATION: 'sentiment.dev'
        }
        sentiment_id = (0, 1)
        for s, n in splits2names.items():
            duplicate_sentences = set()
            if s not in self._preprocessed_files or s not in splits:
                continue
            output_path = self._preprocessed_files[s]
            sentences = []
            labels = []
            for s_i in sentiment_id:
                raw_path = os.path.join(self.raw_dir, f'{n}.{s_i}')
                if not os.path.isfile(raw_path):
                    raise ValueError
                rows = read_lines_from_file(raw_path)
                for r in rows:
                    if r in duplicate_sentences:
                       continue
                    sentences.append(r)
                    labels.append(s_i)
                    duplicate_sentences.add(r)

            output_dir = os.path.split(output_path)[0]
            os.makedirs(output_dir, exist_ok=True)
            dataset = Dataset.from_dict({
                "sentence": sentences,
                "label": labels
            })
            dataset.to_csv(output_path, index=False)

    @property
    def an_sample(self) -> Tuple[str]:
        return "i was sadly mistaken .", 0

    @property
    def input_column_name(self):
        return "sentence"

    @property
    def label_column_name(self):
        return "label"
    
    # def _load_dataset(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT) -> Dict[DatasetSplitType, Dataset]:
    #     result = super()._load_dataset(splits)
    #     self._update_max_length(list(result.values()))
    #     return result

            
    @property
    def max_length(self):
        if not isinstance(self._max_length, int):
            self._max_length = max_length_of_sequences(self, dataset=list(self._dataset.values()))
        return self._max_length

            


