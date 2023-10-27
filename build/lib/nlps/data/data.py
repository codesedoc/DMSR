import argparse
import json
import os
import pickle
from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
from collections import OrderedDict
from enum import Enum
from typing import Tuple, Dict, Callable, Union, Any, List, Iterable, MutableMapping, Optional, Set

import numpy as np
import torch
import wget
import wget
from torch.utils.data.dataset import T_co
from ..argument import ArgumentConfigurator, NECESSARY, ArgumentPool, argument_class, ArgumentParser
from ..argument import DataArgument
from ..argument.argument import PATH_MODE
from ..pipeline import UniversalPort
import datasets
from datasets import Dataset
import pandas as pd
from dataclasses import dataclass, fields, field

from ..utils import singleton
from ..utils.utils import MetaMetrics, SequencePairBatch, MetricComputerProxy

class DatasetSplitType(Enum):
    TEST = 'test'
    TRAIN = 'train'
    VALIDATION = 'validation'

    @classmethod
    @property
    def items(cls):
        return tuple(DatasetSplitType)

    @classmethod
    @property
    def item_values(cls):
        return tuple([i.values for i in cls.items])


ALL_DATASET_SPLIT = (DatasetSplitType.TRAIN, DatasetSplitType.TEST, DatasetSplitType.VALIDATION)


class DataWrapper:
    def __init__(self, content=None):
        self.content = content


class DataContainer(dict):
    def __init__(self, names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for n in names:
            self[n] = list()


class ApplicationDataContainer(DataContainer):
    def __init__(self, *args, **kwargs):
        super(ApplicationDataContainer, self).__init__(['input', 'output_wrapper'], *args, **kwargs)
        self.input = self["input"]
        self.output_wrapper = self["output_wrapper"]

    @property
    def input(self):
        return self["input"]

    @input.setter
    def input(self, value):
        self["input"] = value

    @property
    def output_wrapper(self):
        return self["output_wrapper"]

    @output_wrapper.setter
    def output_wrapper(self, value):
        self["output_wrapper"] = value


class DataCollator:
    def __init__(self, fetch_function, container: DataContainer = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_function = fetch_function
        self.container = container

    def collate(self, data):
        return self.fetch_function(self.container, data)


class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Union[List[Any], Tuple[Any], Dict[str, Union[List[Any], Tuple[Any]]]]):
        if isinstance(samples, dict):
            _samples = [{} for i in range(len(list(samples.values())[0]))]
            for k, v in samples.items():
                if len(v) != len(_samples):
                    raise ValueError

                for i, item in enumerate(_samples):
                    if k in item:
                        raise ValueError
                    item[k] = v[i]
            self.samples = _samples
        else:
            self.samples = samples

        super().__init__()

    def map(self, _callable: Callable):
        for i, s in enumerate(self.samples):
            self.samples[i] = _callable(s)

    def __getitem__(self, index) -> T_co:
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class DataDirCategory(Enum):
    RAW = 'raw'
    PREPROCESSED = 'preprocessed'
    APPLICATION = 'application'


class TaskType(Enum):
    CLASSIFICATION = 'classification'
    GENERATION = 'generation'
    REGRESSIVE = "regressive"


@singleton
class HFDatasetProxy:
    def __init__(self):
        self.dataset_paths = datasets.list_datasets()

    def is_hf_dataset(self, path:Union[str, List[str]]):
        if isinstance(path, str):
            return path in self.dataset_paths
        elif isinstance(path, list):
            return [self.is_hf_dataset(p)for p in path]


class MultiTaskDataset(Dataset):
    _task_column_name = 'task_name'

    def __init__(self, *args, task_names: Tuple = None, **kwargs):
        super().__init__(*args, **kwargs)

        if self.task_column_name not in self.column_names:
            raise ValueError

        if isinstance(task_names, tuple):
            names = set()
            for n in task_names:
                assert isinstance(n, str) and len(n.strip()) > 0
                if n in names:
                    raise ValueError
                names.add(n)
            for n in self["task_name"]:
                if n not in names:
                    raise ValueError
            self._names = task_names
        else:
            names = set()
            for n in self["task_name"]:
                assert isinstance(n, str) and len(n.strip()) > 0
                names.add(n)

            self._names = tuple(names)

    @property
    def task_names(self):
        return self._names

    @classmethod
    @property
    def task_column_name(cls):
        return cls._task_column_name


def merge_datasets(datasets_:Tuple[Union[Dataset, Dict[DatasetSplitType, Dataset]]]) \
        -> Union[MultiTaskDataset, Dict[DatasetSplitType, MultiTaskDataset]]:
    if not isinstance(datasets_, Iterable) or len(datasets_) <= 0:
        raise ValueError
    first_item = datasets_[0]
    if isinstance(first_item, Dataset):
        result:Dict[str, list] = datasets_[0].to_dict()
        for d in datasets_[1:]:
            delete_name = set(result.keys())
            d = d.to_dict()
            for c_m in d.keys():
                if c_m in result.keys():
                    if type(result[c_m][0]) is not type(d[c_m][0]):
                        if not isinstance(result[c_m][0], str):
                            result[c_m] =[f"Stringed_Column:{str(item)}" for item in result[c_m]]

                        if not isinstance(d[c_m][0], str):
                            d[c_m] =[f"Stringed_Column:{str(item)}" for item in d[c_m]]

                    result[c_m].extend(d[c_m])
                    delete_name.remove(c_m)
            for n in delete_name:
                result.pop(n)
        result = MultiTaskDataset.from_dict(result)

    elif isinstance(first_item, dict):
        result: Dict[DatasetSplitType, List] = dict()
        for d in datasets_:
            for k, v in d.items():
                if k not in result:
                    result[k] = list()
                result[k].append(v)

        for t in result.keys():
            dataset = merge_datasets(tuple(result[t]))
            result[t] = dataset
    else:
        raise ValueError
    return result


class Data(ArgumentConfigurator, ABC):
    _abbreviation: str = None
    _metric_name_path: str = None
    _hf_metric: bool = False
    _root_dir: str = None
    _task_type: TaskType = None
    _argument_class = DataArgument

    def __init__(self, name: str = None, pre_load=False, argument=None, local= True):
        super(Data, self).__init__(argument=argument, reset_argument=True)

        Data._root_dir = self.args.data_root_dir

        hf_evaluator_list_cache = os.path.join(ArgumentPool.meta_arguments.cache_dir, 'hf_evaluator_list.pkl')
        if not os.path.isfile(hf_evaluator_list_cache) or ArgumentPool.meta_arguments.force_cache:
            import evaluate
            metrics = evaluate.list_evaluation_modules(module_type="metric")
            with open(hf_evaluator_list_cache, mode='wb') as f:
                pickle.dump(metrics, f)
        else:
            with open(hf_evaluator_list_cache, mode='rb') as f:
                metrics = pickle.load(f)
        self._hf_metric = False
        if self._metric_name_path in metrics:
            self._hf_metric = True
        elif self.args.data_root_dir.startswith("/") and not self._metric_name_path.startswith("/"):
            self._metric_name_path = os.path.join(os.getcwd(), self._metric_name_path)

        if ArgumentPool.meta_arguments.path_mode is PATH_MODE.ABSTRACT:
            self.args.data_root_dir = os.path.join(os.getcwd(), self.args.data_root_dir)
        self._io_port: UniversalPort = UniversalPort()
        self._dataset: Dict[DatasetSplitType, Dataset] = None
        self.name = name if name is not None else self.__class__.__name__
        self._is_local_data = local
        if local:
            if not os.path.isdir(self.data_dir()):
                self.download()
            _preprocessed_dir = self.data_dir(category=DataDirCategory.PREPROCESSED)
            if not os.path.exists(_preprocessed_dir):
                os.system(f'mkdir -p {_preprocessed_dir}')

            self._preprocessed_files: Dict[DatasetSplitType, str] = {s: os.path.join(_preprocessed_dir, f'{s.value}.csv')
                                                                     for s in ALL_DATASET_SPLIT}
            for s in ALL_DATASET_SPLIT:
                output_path = os.path.join(_preprocessed_dir, f'{s.value}.csv')
                if os.path.isfile(output_path) and not self.args.force_preprocess:
                    continue
                self._preprocessed_files[s] = output_path

            self.raw_dir = self.data_dir(category=DataDirCategory.RAW)
            if not os.path.exists(self.raw_dir):
                print(f"Raw data is not exist, deleted broken data dir '{self.data_dir()}'!")
                os.system(f'rm -r {self.data_dir()}')
                raise ValueError

        if pre_load:
            self.load_dataset()
            assert isinstance(self._dataset, dict)

        self.current_approach: str = None
        Data._metric_name_path = None

    @property
    def metric_computer(self):
        return MetricComputerProxy().metric_computer(self.abbreviation, self._load_metric)

    @classmethod
    @property
    def abbreviation(cls) -> str:
        result = cls._abbreviation
        if hasattr(cls, "_config_name"):
            result = os.path.join(result, cls._config_name)
        return result

    @classmethod
    @property
    def metric_name_path(cls) -> str:
        return cls._metric_name_path

    @property
    @abstractmethod
    def input_column_name(self) -> str:
        raise RuntimeError

    @property
    @abstractmethod
    def label_column_name(self) -> str:
        raise RuntimeError

    @property
    def input_name(self) -> str:
        return str(self.input_column_name)

    @property
    def label_name(self) -> str:
        return 'labels'

    @classmethod
    @property
    def root_dir(cls):
        return cls._root_dir

    def extract_input_label_from_samples(self, samples: Dataset, *args, **kwargs) -> Tuple[Union[SequencePairBatch, List[Any]], Optional[List[Any]]]:
        if isinstance(samples, Dataset):
            names = samples.column_names
        elif isinstance(samples, MutableMapping):
            names = samples.keys()
        else:
            raise ValueError

        if isinstance(self.input_column_name, str):
            input_ = samples[self.input_column_name]
        elif isinstance(self.input_column_name, Iterable):
            input_ = tuple(samples[i] for i in self.input_column_name)
            if len(self.input_column_name) == 2:
                input_ = SequencePairBatch(*input_)
        else:
            raise ValueError

        if self.label_column_name in names:
            return input_, samples[self.label_column_name]
        else:
            return input_, None

    @abstractmethod
    def _preprocess(self,  split: DatasetSplitType, *args, **kwargs):
        ...

    @classmethod
    def data_dir(cls, name: str = None, category: DataDirCategory = None, **kwargs):
        if name is None:
            name = cls.abbreviation
        result = os.path.join(cls.root_dir, name)
        if category is not None:
            result = os.path.join(result, category.value)
            if category is DataDirCategory.APPLICATION:
                result = os.path.join(result, kwargs.get('approach_name', ArgumentPool.meta_arguments.approach))
        return result

    @classmethod
    def download(cls):
        name = f"{cls.abbreviation}.tar.gz"
        package_path = os.path.join(cls.root_dir, name)
        package_dir = os.path.split(package_path)[0]
        try:
            cls._download()
        except:
            if os.path.exists(package_dir):
                os.system(f'rm -r {package_dir}')
            raise

    @classmethod
    def _download(cls):

        if not os.path.isdir(cls._root_dir):
            assert not os.system(f'mkdir -p {cls._root_dir}')

        name = f"{cls.abbreviation}.tar.gz"
        package_path = os.path.join(cls.root_dir, name)
        package_dir = os.path.split(package_path)[0]
        if not os.path.isdir(package_dir):
            os.system(f"mkdir -p {package_dir}")
        if os.path.isfile(package_path):
            print(f'Data package ({package_path}) has been downloaded!')
            assert os.system(f'tar -xzf {package_path} -C {package_dir}') == 0
            print("Unpackaging is Successful!")
            return 0

        ns_lookup_flag = os.system('nslookup -timeout=3 file.pretrain.beta')
        if ns_lookup_flag == 0:
            with os.popen("nslookup -timeout=3 file.pretrain.beta | grep Address | awk  'END {print $NF}'",
                          mode='r') as p:
                ip_address = p.read()[:-1]
                p.close()
            url = f"http://file.pretrain.beta:8084/{os.path.basename(name)}"

            try:
                wget.download(url=url, out=package_path)
            except Exception:
                raise ValueError(f"Can't download data from '{url}'")
            print(f"\nDownloaded data from '{url}'")

            assert os.system(f'tar -xzf {package_path} -C {package_dir}') == 0
        else:
            try:
                git_server = os.environ['GIT_SERVER']
                project = os.environ['PROJECT']
                user = os.environ['USER']
            except Exception:
                raise ValueError(f"Lack of environment variables ('GIT_SERVER' or 'PROJECT'), failed!")
            if package_path.startswith("/"):
                scp_source = f'{user}@{git_server}:{package_path}'
            else:
                scp_source = os.path.join(f'{user}@{git_server}:$HOME', project, package_path)
            try:
                assert os.system(f'scp {scp_source} {package_dir}') == 0
            except Exception:
                # print()
                print("server_ip: " + git_server)
                print(f'Coping data from {scp_source}:')
                breakpoint()
                raise ValueError(f"Can't copy from {scp_source}, failed!")
            assert os.system(f'tar -xzf {package_path} -C {package_dir}') == 0

            print("Successful!")

    def _dataset_for_application(self, runtime: Dict[str, Any], *args, **kwargs) -> Dataset:
        dataset = self.dataset(DatasetSplitType.VALIDATION)[0]
        column_names = dataset.column_names
        remove_column_names = [c for c in column_names if c != self.input_column_name]
        if len(remove_column_names) > 0:
            dataset.remove_columns(remove_column_names)
        dataset = dataset.rename_column(original_column_name=self.input_column_name, new_column_name="input")
        return dataset

    def application_dataset_collate(self, runtime: Dict[str, Any], *args, **kwargs) -> Dataset:
        dataset = self._dataset_for_application(runtime, *args, **kwargs)
        runtime["dataset"] = dataset

    def _application_finish_call_back(self, runtime: Dict[str, Any]):
        pass

    def application_finish_call_back(self, runtime: Dict[str, Any]):
        dataset = runtime.get("dataset", None)
        if not isinstance(dataset, Dataset):
            return None

        self._application_finish_call_back(runtime)

    def dataset(self, split: Union[None, Tuple[DatasetSplitType], DatasetSplitType] = ALL_DATASET_SPLIT) -> Union[Dataset, Tuple[Dataset]]:
        dataset = self.load_dataset(split)
        return dataset

    def dataset_map(self, map_function: Callable, splits: Tuple[DatasetSplitType] = (DatasetSplitType.TRAIN, DatasetSplitType.VALIDATION)):
        result: List[Dataset] = list()
        if splits is None or len(splits) == 0:
            return result

        dataset = self.dataset(splits)
        for s, d in zip(splits, dataset):
            self._dataset[s] = d.map(map_function, batched=True)
            result.append(self._dataset[s])
        return tuple(result)

    def _load_metric(self):
        # import evaluate
        # metrics = evaluate.list_evaluation_modules()
        pwd = __file__
        current_dir = os.path.split(pwd)[0]
        # if self._metric_name_path in metrics:
        if self._hf_metric:
            if isinstance(self, HFData):
                assert self._metric_name_path in ("glue",)
                metric = datasets.load_metric(path=self._metric_name_path, config_name=self._config_name)
            else:
                metric = datasets.load_metric(path=self._metric_name_path)
        elif self._metric_name_path.startswith('/'):
            try:
                assert os.path.exists(self._metric_name_path)
            except AssertionError:
                print(f"Path '{self._metric_name_path}' do not exist!")
                raise AssertionError
            metric = datasets.load_metric(self._metric_name_path)
        elif self._metric_name_path is None:
            raise ValueError(f"{self}: did not set '_metric_name_path'!")
        elif os.path.exists(os.path.join(current_dir, 'metrics', self._metric_name_path)):
            metric = datasets.load_metric(os.path.join(current_dir, 'metrics', self._metric_name_path))
        # elif self.args.data_root_dir is None:
        #     raise ValueError(f"{self}: did not set 'root_dir'!")
        elif os.path.exists(os.path.join(self.args.data_root_dir, self.abbreviation,self._metric_name_path)):
            metric = datasets.load_metric(os.path.join(self.args.data_root_dir, self.abbreviation,self._metric_name_path))

        else:
            print(current_dir)
            breakpoint()
            print(os.path.isdir(os.path.join(self.args.data_root_dir, self.abbreviation,self._metric_name_path)))
            print(os.path.join(self.args.data_root_dir, self.abbreviation,self._metric_name_path))
            raise ValueError
        return metric

    def _compute_metrics(self, predictions: np.ndarray, labels: np.ndarray, inputs:  np.ndarray = None, *args, **kwargs):
        _metric_computer = self.metric_computer
        if not self._hf_metric:
            kwargs["inputs"] = inputs

        return _metric_computer.compute(*args, predictions=predictions, references=labels, **kwargs)

    def _update_meta_metrics(self, metrics: dict, predictions: np.ndarray, labels: np.ndarray, inputs:  np.ndarray = None,
                             prediction_to_output: Callable = None, *args, **kwargs):
        if labels is None:
            labels = ['null'] * len(predictions)

        examples = [
            {
                "order": i,
                self.input_name: str(inputs[i]) if inputs is not None else None,
                self.label_column_name: str(prediction_to_output(pair[0])),
                "prediction": str(prediction_to_output(pair[1])),
            }
            for i, pair in enumerate(zip(labels, predictions))
        ]
        if "meta_metrics" in metrics:
            meta_metrics: MetaMetrics = metrics["meta_metrics"]
            if isinstance(meta_metrics.examples, list):
                start_index = len(meta_metrics.examples)
                for e in examples:
                    e["order"] += start_index
                meta_metrics.examples.extend(examples)
            else:
                meta_metrics.examples = examples
        else:
            meta_metrics: MetaMetrics = MetaMetrics(examples)
        return meta_metrics

    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray, inputs:  np.ndarray = None,
                        prediction_to_output: Callable = None, *args, **kwargs):

        if not isinstance(self._metric_name_path, str) or self._metric_name_path == '':
            raise ValueError

        result = dict(self._compute_metrics(predictions, labels, inputs))
        # breakpoint()
        if ArgumentPool["tune_hp"]:
            return result

        if not isinstance(prediction_to_output, Callable):
            prediction_to_output = lambda x: x

        result["meta_metrics"] = self._update_meta_metrics(result, predictions, labels, inputs, prediction_to_output)
        return result

    @property
    def _field_name2value_factory(self) -> Optional[Dict[str, Callable]]:
        return None

    def _load_dataset(self, split: DatasetSplitType, *args, **kwargs) -> Dataset:
        print(f"**** Loading {split.value} set from: {self.abbreviation} ****")
        assert isinstance(split, DatasetSplitType)
        data_file = self._preprocessed_files.get(split, None)
        if not self._is_local_data or not os.path.isfile(data_file) or self.args.force_preprocess:
            self._preprocess(split, *args, **kwargs)

        extension = os.path.splitext(data_file)[-1][1:]
        try:
            result = datasets.load_dataset(extension, data_files={split.value: data_file}, split=split.value)
            assert isinstance(result, Dataset)
        except Exception:
            raise RuntimeError(f"Load data split '{split.value} from local file ('{data_file}') failed")

        return result

    def _modify_dataset(self, dataset: Dataset):
        assert isinstance(dataset, Dataset)
        result = dataset
        if self.args.smoke_test:
            result = Dataset.from_dict(result[:min(16, len(result))])
        name2type = self._field_name2value_factory
        if isinstance(name2type, dict):
            result = result.map(lambda sample_dict: {name: [name2type[name](s) if name in name2type else s
                                                            for s in samples]
                                                     for name, samples in sample_dict.items()},
                                batched=True)
        return result

    def load_dataset(self, split: Union[DatasetSplitType, Tuple[DatasetSplitType]] = (DatasetSplitType.TRAIN, DatasetSplitType.VALIDATION),  *args, **kwargs) -> Union[Dataset, Tuple[DatasetSplitType, Dataset]]:
        if isinstance(split, Iterable):
            result = []
            for s in split:
                assert isinstance(s, DatasetSplitType)
                result.append(self.load_dataset(s))
            return type(split)(result)

        elif isinstance(split, DatasetSplitType):
            if not isinstance(self._dataset, dict):
                self._dataset = dict()
            if split not in self._dataset:
                d = self._load_dataset(split)
                d = self._modify_dataset(d)
                assert isinstance(d, Dataset)
                self._dataset[split] = d
            result = self._dataset[split]

        elif isinstance(split, None):
            result = None
        else:
            raise ValueError

        return result

    def __getitem__(self, item):
        if self._dataset is None:
            return self
        for k in self._dataset.keys():
            if not isinstance(item, slice):
                item = slice(item)
            self._dataset[k] = Dataset.from_dict(self._dataset[k][item])
        return self

    @property
    def target_metric(self):
        return {
            "name": "loss",
            "direction": -1,
        }

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def _category_names(self) -> List[str]:
        raise ValueError

    @property
    def category_names(self) -> List[str]:
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError
        return self._category_names()

    def _label_scalar(self) -> Tuple[float]:
        raise ValueError

    @property
    def label_scalar(self) -> Tuple[float]:
        if self.task_type != TaskType.REGRESSIVE:
            raise ValueError
        return self._label_scalar()

    @property
    def metric_names(self):
        an_sample = self.an_sample
        result = tuple(self.compute_metrics(predictions=[an_sample[1]], labels=[an_sample[1]], inputs=[an_sample[0]]).keys())
        return {"names": result, "target": self.target_metric}

    @property
    @abstractmethod
    def an_sample(self) -> Tuple[Any]:
        ...

    @classmethod
    @property
    def argument_class(cls):
        return cls._argument_class

    @classmethod
    def collect_argument(cls):
        ArgumentPool.push(arg_class=cls.argument_class, revise_cls=True)

    def assign_argument(self):
        self.args: DataArgument = ArgumentPool.pop(self.argument_class)

    # Name2DataClass: Dict[str, type] = dict()


class HFData(Data, ABC):
    _path = None
    _config_name = None

    def __init__(self):
        if self._path is None:
            self._path = self.abbreviation

        hfp = HFDatasetProxy()
        if not hfp.is_hf_dataset(self._path):
            raise ValueError

        super(HFData, self).__init__(local=False)

    def _preprocess(self, *args, **kwargs):
        pass

    @classmethod
    def _download(cls):
        pass

    def _load_dataset(self, split: DatasetSplitType, *args, **kwargs) -> Dataset:
        return datasets.load_dataset(path=self._path, name=self._config_name, split=split.value)


# def _name2dataclass_path():
#     pwd = __file__
#     current_dir = os.path.split(pwd)[0]
#     file_path = os.path.join(current_dir, "name2dataclass.pkl")
#     return file_path


# def load_name2dataclass(file_path: str = None):
#     global Name2DataClass
#     if not isinstance(file_path, str):
#         file_path = _name2dataclass_path()
#
#     if os.path.isfile(file_path):
#         with open(file_path, mode="rb") as f:
#             Name2DataClass = pickle.load(f)
#     else:
#         Name2DataClass = dict()
#
#
# def save_name2dataclass(file_path: str = None):
#     global Name2DataClass
#     if not isinstance(file_path, str):
#         file_path = _name2dataclass_path()
#     with open(file_path, mode="wb") as f:
#         pickle.dump(Name2DataClass, f)


Name2DataClass: Dict[str, type] = dict()
# load_name2dataclass()


def data_register(cls: Data):
    if not issubclass(cls, Data) or cls.abbreviation is None:
        raise ValueError
    abbreviation = cls.abbreviation
    Name2DataClass[abbreviation] = cls
    return cls
