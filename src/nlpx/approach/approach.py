import inspect
import os
import pickle
import socket
from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
from dataclasses import field
from enum import Enum
from functools import wraps
from typing import Callable, Dict, Any, Union, Optional, List, Tuple, Iterable
import datasets
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from .loss import LOSS_FN_NAME2CLASS
from ..argument import ArgumentConfigurator, ExternalArgument, ArgumentPool, argument_class, ApproachArgument, NECESSARY, \
    Argument, ArgumentType, ModelArgument
from ..argument.argument import PATH_MODE
from ..data import Data, DatasetSplitType, GeneralDataset, Name2DataClass
from ..data.data import DataDirCategory, DataContainer, DataWrapper
from ..pipeline import UniversalPort
from ..utils.declare import UNKNOWN


class Approach(ArgumentConfigurator, ABC):
    _abbreviation = UNKNOWN
    _type = "approach"
    _argument_class = ApproachArgument
    _model_argument_class = ModelArgument

    def __init__(self, model: Callable = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_args()
        self._io_port: UniversalPort = None
        self._model = model
        self.precessing_data: Data = None
        self._process_data_split: Tuple[DatasetSplitType] = DatasetSplitType.TEST,
        # self._model_init = self.__model_init__wrapper__(self._model_init)

    def _update_args(self):
        self._reset_argument()
        meta_arguments = ArgumentPool.meta_arguments
        if meta_arguments.path_mode is PATH_MODE.ABSTRACT:
            self.args.output_dir = os.path.join(os.getcwd(), self.args.output_dir)
        data_path_component = meta_arguments.dataset if not hasattr(self, "precessing_data") else self.precessing_data.abbreviation
        self.args.output_dir = os.path.join(self.args.output_dir, self.path_component, data_path_component,
                                            socket.gethostname(), self.model_args.model_name)

    # def __model_init__wrapper__(self, method):
    #     @wraps(method)
    #     def wrapper(*args, **kwargs):
    #         model = method(*args, **kwargs)
    #         setattr(model, "approach", self)
    #         return model
    #     return wrapper

    @abstractmethod
    def _model_init(self, *args, **kwargs):
        ...

    def model_init(self, *args, **kwargs):
        model = self._model_init(*args, **kwargs)
        # setattr(model, 'affiliation', self)
        return model

    @classmethod
    @property
    def abbreviation(cls) -> str:
        return cls._abbreviation

    @property
    def path_component(self):
        return self._abbreviation

    @classmethod
    @property
    def argument_class(cls):
        return cls._argument_class

    @classmethod
    @property
    def model_argument_class(cls):
        return cls._model_argument_class

    @property
    def io_port(self):
        return self._io_port

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise ValueError
        return self._model

    @model.setter
    def model(self, value):
        assert value is None or isinstance(value, torch.nn.Module)
        self._model = value

    @property
    def processing_data(self):
        return self.precessing_data

    @processing_data.setter
    def processing_data(self, value):
        self.precessing_data = value

    @property
    def compute_metric(self):
        return self.precessing_data.compute_metrics

    def _request_datasets(self, data: Data = None, *args, **kwargs) -> Optional[Tuple[Dataset]]:
        data = data if isinstance(data, Data) else self.precessing_data
        data_splits = kwargs.get('data_splits', self._process_data_split)
        if len(data_splits) > 0:
            data.load_dataset(data_splits)
            # if self.args.smoke_test:
            #     data = data[:16]
            precessing_data = self.precessing_data
            self.precessing_data = data
            dataset = data.dataset_map(self._preprocess, splits=data_splits)
            self.precessing_data = precessing_data
        else:
            dataset = None
        return dataset

    def _request_datasets_to_dict(self, data: Data = None, dataset: Tuple[Dataset] = None, **kwargs) -> Optional[Dict[DatasetSplitType, Dataset]]:
        data_splits = kwargs.get('data_splits', self._process_data_split)
        if not (isinstance(dataset, tuple) and len(dataset) > 1 and isinstance(dataset[0], Dataset)):
            dataset = self._request_datasets(data = data, data_splits = data_splits)
        assert len(data_splits) == len(dataset)
        return {s: d for s, d in zip(data_splits, dataset)}

    def _pre_call_hock(self):
        # self._update_args()
        pass

    @property
    def trail_name(self):
        return self.abbreviation

    def __call__(self, data: Data, *args, **kwargs):
        self.precessing_data = data
        self._pre_call_hock()
        if self.args.tune_hp:
            dataset = self._request_datasets_to_dict(data_splits=[DatasetSplitType.TRAIN, DatasetSplitType.VALIDATION])
            self.tune_hyperparameter(dataset)
        else:
            dataset = self._request_datasets_to_dict(**kwargs)
            self._model = self._model_init()
            self._process(dataset)
            self._post_process(dataset)

        self.precessing_data = None
        if self._model is not None and hasattr(self._model, "approach"):
            delattr(self._model, "approach")

    def _preprocess(self, samples: Union[Dataset, Dict[str,Any]]):
        return samples

    @abstractmethod
    def _process(self, dataset: Dict[DatasetSplitType, Union[None,datasets.Dataset]], *args, **kwargs):
        ...

    @abstractmethod
    def _post_process(self, dataset: Dict[DatasetSplitType, Union[None,datasets.Dataset]], *args, **kwargs):
        ...

    def _default_application_launcher(self):
        from ..utils.application import Application
        print(f"**** Conduct Application for data '{self.precessing_data.abbreviation}' ****")
        application = Application(data=Name2DataClass[self.precessing_data.args.application_dataset](), approach=self)
        application.run()

    @abstractmethod
    def _application(self, data: Data, runtime: Dict[str, Any]):
        ...

    def application(self, data: Data, runtime: Dict[str, Any]):
        dataset = runtime.get("dataset", None)
        if not isinstance(dataset, Dataset):
            raise ValueError
        result = self._application(data, runtime)
        return result

    @classmethod
    def collect_argument(cls, *arg, **kwargs):
        ArgumentPool.push(arg_class=cls.argument_class)
        ArgumentPool.push(arg_class=cls.model_argument_class)

    def assign_argument(self):
        self.args: ApproachArgument = ArgumentPool.pop(self.argument_class)
        self.model_args: ModelArgument = ArgumentPool.pop(self.model_argument_class)

    @abstractmethod
    def metric_names(self, *args, data: Data = None, **kwargs):
        ...

    @abstractmethod
    def target_metric_for_tuning_hp(self, *args, data: Data = None, **kwargs):
        ...

    @abstractmethod
    def tune_hyperparameter(self, dataset:Dict[DatasetSplitType, Dataset], *args, **kwargs):
        ...

    def release_model(self, model:torch.nn.Module = None):
        if model is None:
            model = self._model
        if model is not None:
            model.to_empty(device=torch.device('cpu'))


Name2ApproachClass: Dict[str, type] = dict()


def approach_register(cls: Approach):

    if not issubclass(cls, Approach) or cls.abbreviation is None:
        raise ValueError

    for m in inspect.getmembers(cls):
        if m[1] == UNKNOWN:
            raise ValueError(f"Have to set a value to attribute '{m[0]}' for class {cls.__name__}, "
                             f"please declare it.")

    abbreviation = cls.abbreviation
    Name2ApproachClass[abbreviation] = cls
    return cls


@argument_class
class NeuralNetWorkArgument(ApproachArgument):
    gpu_per_trial: float = field(
        default=1,
        metadata={
            'help': "Specify the gpu resource for tuning hyperparameter."
        }
    )

    loss_fn: str = field(
        default=None,
        metadata={
            'help': "Specify one of the loss functions",
            'choices': f"{str(list[LOSS_FN_NAME2CLASS])}"
        }
    )


@argument_class
class NNModelArgument(ModelArgument):
    pass


class NeuralNetWork(Approach, ABC):
    _type = 'neural_network'
    _argument_class = NeuralNetWorkArgument
    _model_argument_class = NNModelArgument

    def __init__(self, *args, **kwargs):
        super(NeuralNetWork, self).__init__(*args, **kwargs)





