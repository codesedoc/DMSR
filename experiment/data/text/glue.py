import os
from abc import ABC
from typing import Dict, Any, Tuple

from datasets import Dataset

from nlpx.data import HFData, DatasetSplitType, TaskType, data_register
from nlpx.data.data import ALL_DATASET_SPLIT
from nlpx.utils.utils import max_length_of_sequences, SequencePairBatch


class GLUE(HFData, ABC):
    _path = "glue"
    _abbreviation = 'glue'
    _metric_name_path = "glue"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_length = None

    # def _load_dataset(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT) -> Dict[DatasetSplitType, Dataset]:
    #     result = super()._load_dataset(splits)
    #     self._max_length = max_length_of_sequences(list(result.values()))
    #     return result

    @property
    def max_length(self):
        if not isinstance(self._max_length, int):
            self._max_length = max_length_of_sequences(self, dataset=list(self._dataset.values()))
        return self._max_length


@data_register
class STSB(GLUE):
    _task_type: TaskType = TaskType.REGRESSIVE
    _config_name = "stsb"

    @property
    def input_column_name(self):
        return 'sentence1', 'sentence2'

    @property
    def label_column_name(self):
        return 'label'

    @property
    def an_sample(self) -> Tuple[Any]:
        return ("A plane is taking off.", "An air plane is taking off."), 5.0

    def _label_scalar(self):
        return 0, 5

    def extract_input_label_from_samples(self, samples: Dataset, *args, **kwargs):
        input_, label = super().extract_input_label_from_samples(samples, *args, **kwargs)
        if not isinstance(input_, SequencePairBatch):
            input_ = SequencePairBatch(*input_)
        return input_, label

    def _load_dataset(self, split: DatasetSplitType, *args, **kwargs) -> Dataset:
        dataset = super()._load_dataset(split, *args, **kwargs)
        delete_label = False
        for l in dataset["label"]:
            if l < 0 or l >5:
                delete_label = True
                break
        if delete_label:
            dataset = dataset.remove_columns("label")
        return dataset


if __name__ == '__main__':
    pass
