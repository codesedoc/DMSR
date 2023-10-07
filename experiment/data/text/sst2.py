from typing import Tuple, Any, List, Dict

from datasets import Dataset

from experiment.data.text.glue import GLUE
from src.nlps.data import data_register, TaskType, DatasetSplitType
from src.nlps.data.data import ALL_DATASET_SPLIT


@data_register
class SST2(GLUE):
    _config_name = "sst2"
    _task_type = TaskType.CLASSIFICATION

    @property
    def label_column_name(self):
        return "label"

    @property
    def input_column_name(self):
        return "sentence"

    @property
    def an_sample(self) -> Tuple[Any]:
        return "hide new secretions from the parental units", 0

    def extract_input_label_from_samples(self, samples: Dataset, *args, **kwargs):
        if "label" in samples:
            return samples["sentence"], samples["label"]
        else:
            return samples["sentence"], None

    def _class_names(self) -> List[str]:
        return ["positive", "negative"]

    def _load_dataset(self, splits: Tuple[DatasetSplitType] = ALL_DATASET_SPLIT):
        dataset: Dict[DatasetSplitType, Dataset] = super()._load_dataset(splits)
        for t, d in dataset.items():
            delete_label = False
            for l in d["label"]:
                if l != 0 and l != 1:
                    delete_label = True
                    break
            if delete_label:
                dataset[t] = d.remove_columns("label")
        return dataset
