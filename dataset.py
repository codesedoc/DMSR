from enum import Enum, StrEnum, auto
import math
from pathlib import Path
from dataclasses import field, dataclass
from evaluate import Evaluator
import evaluate
import datasets
from datasets import Dataset
from evaluate import EvaluationModule
from nlpe import ArgumentPool, EvaluatorProxy, DatasetProxy, DatasetSplitCategory, Pool, Data, TextData, Text
from nlpe.utils import Glossary, global_logger
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any, MutableMapping, Callable
import os
import pandas as pd

from model import VariantGlossaryEnum
from utils import GlossaryEnum


def _get_split_path(dataset_dir: Path, split: DatasetSplitCategory) -> Path:
    assert isinstance(dataset_dir, Path)
    assert isinstance(split, DatasetSplitCategory)
    return Path(dataset_dir, "vanilla", f'{split}.csv')


def _load_dataset(file_path: Path) -> Dataset:
    result = datasets.load_dataset("csv", data_files=str(file_path))["train"]
    if ArgumentPool().meta_argument["debug"]:
        dataset_list = result.to_list()
        dataset_list = dataset_list[ : min(10, len(dataset_list))]
        result = Dataset.from_list(dataset_list)
    return result
    
def _pseduo_dataset_load_call(proxy: DatasetProxy, split: DatasetSplitCategory, *args, **kwargs):
    assert proxy.raw_dir != Path(ArgumentPool().meta_argument["dataset_raw_dir"])
    split_path: Path = _get_split_path(proxy.raw_dir.parent, split)
    if not split_path.is_file():
        split2rate: Dict[DatasetSplitCategory, float] = {
            DatasetSplitCategory.TRAIN: (0, 0.8),
            DatasetSplitCategory.TEST: (0.8, 0.9),
            DatasetSplitCategory.VALIDATION: (0.9, 1.0)
        }
        dataset = datasets.load_dataset("csv", data_dir=proxy.raw_dir)["train"]
        data_size = len(dataset)
        span = tuple(map(lambda x: math.floor(x * data_size), split2rate[split]))
        split_dataset = Dataset.from_dict(dataset[span[0]:span[1]])
        split_path.parent.mkdir(exist_ok=True)
        if split != DatasetSplitCategory.TRAIN:
            unique_input = set()
            final_samples = []
            for sample in split_dataset:
                if sample["input"] in unique_input:
                    continue
                unique_input.add(sample["input"])
                final_samples.append(sample)
            if len(final_samples) > 0:
                Dataset.from_list(final_samples).to_csv(split_path, index=False)
        else:
            split_dataset.to_csv(split_path, index=False)
        result = split_dataset
    else:
        result = _load_dataset(split_path)
    if InputColumnName not in result.column_names:
        result = result.rename_column("input", InputColumnName)
    if LabelColumnName not in result.column_names:
        result = result.rename_column("reference", LabelColumnName)
    return result


def _ppf_load_call(proxy: DatasetProxy, split: DatasetSplitCategory, *args, **kwargs) -> Dataset:
    split2name: Dict[str, DatasetSplitCategory] = {
            DatasetSplitCategory.TRAIN: 'wholetrain.csv',
            DatasetSplitCategory.TEST: 'wholetest.csv',
            DatasetSplitCategory.VALIDATION: 'wholedev.csv'
    }
    raw_path = Path(proxy.raw_dir, split2name[split])
    if not raw_path.is_file():
        logger = global_logger()
        logger.error(f"Raw path of dataset ({str(proxy.glossary)}) is not file, its location is '{raw_path}'). It can be specified by optional argument '--dataset_raw_dir'")
        raise ValueError
    split_path = _get_split_path(dataset_raw_dir(DatasetGlossaryEnum.PPF.value), split)
    if raw_path != split_path and (not split_path.is_file()):
        split_path.parent.mkdir(exist_ok=True)
        split_path.write_bytes(raw_path.read_bytes())
    result = _load_dataset(split_path)
    if InputColumnName not in result.column_names:
        result = result.rename_column("original_text", InputColumnName)
    if LabelColumnName not in result.column_names:
        result = result.rename_column("reframed_text", LabelColumnName)
    return result
            

PPF_EVALUATOR: EvaluationModule = evaluate.load(str(Path("storage", "metric", "ppf")), seed=1234)


def dataset_raw_dir(glossary: Glossary) -> Path:
    assert isinstance(glossary, Glossary)
    if str(glossary) == ArgumentPool().meta_argument["dataset"]:
        return Path(ArgumentPool().meta_argument["dataset_raw_dir"])
    return Path("storage", "dataset", str(glossary), "raw")
    
@dataclass    
class DatasetInfo:
    _glossary: Glossary
    _proxy: DatasetProxy = None
    _evaluator: Evaluator = None
    
    def __post_init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self._proxy, DatasetProxy):
            self._proxy: Optional[DatasetProxy] =  DatasetProxy(dataset_type=Dataset, 
                                                                load_dataset_call=DatasetGlossary2LoadCall[self.glossary], 
                                                                dump_dataset_call=None, 
                                                                raw_dir=dataset_raw_dir(self.glossary), 
                                                                glossary=self.glossary)
        if not isinstance(self._evaluator, Evaluator):
            self._evaluator: Optional[Evaluator] = PPF_EVALUATOR
        
    @property
    def load_dataset_call(self) -> Callable:
        if self.glossary in set(DatasetGlossaryEnum.pseudo_glossaries()):
            return _pseduo_dataset_load_call
        elif self.glossary == DatasetGlossaryEnum.PPF.value:
            return _ppf_load_call
        else:
            raise ValueError
        
    @property
    def glossary(self):
        return self._glossary
    
    @property
    def proxy(self) -> DatasetProxy:
       return self._proxy

    @property
    def evaluator(self) -> EvaluationModule:
        return self._evaluator

    def _map_dataset_to_text_list(self, dataset: Dataset) -> List[Text]:
        result = []
        for name in [InputColumnName, LabelColumnName]:
            for t in dataset[name]:
                assert isinstance(t, str)
                result.append(Text(t))
        return result
        
    def to_data(self) -> TextData:
        return TextData(
            dataset_proxy=self.proxy, 
            map_dataset_to_text_list=self._map_dataset_to_text_list,
            evaluator_proxy=EvaluatorProxy(compute_call=self.evaluator.compute, glossary=self.glossary))
    
    @property
    def input_colume(self):
        pass

class DatasetGlossaryEnum(GlossaryEnum):
    MSCOCO2PTR = auto()
    YELP2PTR = auto()
    PPF =  auto()

    @classmethod
    def pseudo_glossaries(cls):
        return [cls.MSCOCO2PTR.value, cls.YELP2PTR.value]
    
    @classmethod
    def is_pseduo(cls, glossary: Glossary) -> bool:
        if glossary.value in set(cls.pseudo_glossaries()):
            return True
        else:
            return False
    
    @staticmethod
    def id(glossary: Glossary = None) -> int:
        assert isinstance(glossary, Glossary)
        return hash(glossary)
        

DatasetGlossary2LoadCall: Dict[Glossary, Callable] = {
    DatasetGlossaryEnum.MSCOCO2PTR.value: _pseduo_dataset_load_call,
    DatasetGlossaryEnum.YELP2PTR.value: _pseduo_dataset_load_call,
    DatasetGlossaryEnum.PPF.value: _ppf_load_call,
}


DatasetGlossaryId2VariantGlossary: Dict[int, Glossary] = {
    DatasetGlossaryEnum.id(DatasetGlossaryEnum.MSCOCO2PTR.value): VariantGlossaryEnum.PG.value,
    DatasetGlossaryEnum.id(DatasetGlossaryEnum.YELP2PTR.value): VariantGlossaryEnum.ST.value,
}

InputColumnName = "input"
LabelColumnName = "reference"
GlossaryIDColumnName = "glossary_id"
# DatasetGlossary2InputColumnName: Dict[Glossary, str] = {
#     DatasetGlossaryEnum.MSCOCO2PTR.value: "input",
#     DatasetGlossaryEnum.YELP2PTR.value: "input",
#     DatasetGlossaryEnum.PPF.value: _ppf_load_call,
# } 
      
#     @classmethod
#     def get_load_dataset_call(cls, glossary: Glossary) -> Callable:
#         if glossary.value in set(cls.pseudo_glossaries()):
#             return _pseduo_dataset_load_call
#         elif glossary == cls.PPF:
#             return _ppf_load_call
#         else:
#             raise ValueError
        
#     @classmethod
#     def text_column_names(cls, glossary: Glossary) -> List[str]:
#         if glossary.value in set(cls.pseudo_glossaries()):
#             return ["input", "reference"]
#         elif glossary == cls.PPF:
#             return ["original_text", "reframed_text"]
#         else:
#             raise ValueError


def merge_datasets(dataset_infos: List[DatasetInfo]) -> DatasetInfo:
    if not isinstance(dataset_infos, Iterable) or len(dataset_infos) <= 1:
        raise ValueError
    split2dataset: Dict[DatasetSplitCategory, Dataset] = dict()
    for s in DatasetSplitCategory.all:
        dataset_list = []
        glossary_column = []
        for d_i in dataset_infos:
            assert isinstance(d_i, DatasetInfo)
            tmp_dataset = d_i.proxy.load_dataset(s)
            dataset_list.append(tmp_dataset)
            glossary_column.extend([DatasetGlossaryEnum.id(d_i.glossary)]*len(tmp_dataset))
        tmp_table:Dict[str, list] = dataset_list[0].to_dict()
        for d in dataset_list[1:]:
            delete_name = set(tmp_table.keys())
            d = d.to_dict()
            for c_m in d.keys():
                if c_m in tmp_table.keys():
                    if type(tmp_table[c_m][0]) is not type(d[c_m][0]):
                        if not isinstance(tmp_table[c_m][0], str):
                            tmp_table[c_m] =[f"Stringed_Column:{str(item)}" for item in tmp_table[c_m]]

                        if not isinstance(d[c_m][0], str):
                            d[c_m] =[f"Stringed_Column:{str(item)}" for item in d[c_m]]

                    tmp_table[c_m].extend(d[c_m])
                    delete_name.remove(c_m)
            for n in delete_name:
                tmp_table.pop(n)
        dataset = Dataset.from_dict(tmp_table)  
        split2dataset[s] = dataset.add_column(name=GlossaryIDColumnName, column=glossary_column)  
    glossary = Glossary("_".join(str(d_i.glossary) for d_i in dataset_infos), force=True)  
    result = DatasetInfo(
        glossary,
        DatasetProxy(dataset_type=Dataset, load_dataset_call=get_load_dataset_call(split2dataset), dump_dataset_call=None, glossary=glossary),
        dataset_infos[0].evaluator
    )
    return result
 
 
def get_load_dataset_call(split2dataset):
    assert isinstance(split2dataset, Dict)
    for s in DatasetSplitCategory.all:
        assert s in split2dataset
    return lambda proxy, split, *args, **kwargs: split2dataset[split]