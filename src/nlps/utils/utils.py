import json
import os.path
from dataclasses import dataclass
from functools import wraps
import random
from typing import Any, Callable, Dict, List, Optional, Iterable, Union

from datasets import Dataset
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch

from .design_patterns import singleton


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

def class_name_chain(cls: type):
    supers = cls.__mro__
    names = [c.__name__ for c in supers]
    return '-->'.join(names)


def jsonable(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = jsonable(v)
        return obj
    else:
        result = obj
        try:
            json.dumps(obj)
        except:
            result = str(obj)
        return result


def _forward_wrapper(model, func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(model, *args, **kwargs)
    return wrapper


def refer_forward(model: torch.nn.Module = None, func=None):
    if not isinstance(model, torch.nn.Module):
        return None
    if not isinstance(func, Callable):
        return None
    forward_wrapper = _forward_wrapper(model, func)

    setattr(model, 'override_forward', model.forward)
    setattr(model, 'forward', forward_wrapper)


def visualize_embeddings(embeddings: np.ndarray, color_list, name=None, output_dir:str=None, show:bool=False, f_size=(6.4, 6.4)):
    assert len(embeddings.shape) == 2
    default_perplexity = 30
    tsne = TSNE(n_components=2, perplexity=min(default_perplexity, len(embeddings)-1))
    if isinstance(output_dir, str) and len(output_dir) > 0:
        if not os.path.isdir(output_dir):
            os.system(f"mkdir -p '{output_dir}'")

    result = tsne.fit_transform(embeddings)
    plt.figure(figsize=f_size)
    plt.subplot()
    if name is None or not isinstance(name, str):
        name = "embedding"
    else:
        name += "-embedding"
    plt.scatter(result[:, 0], result[:, 1], c=color_list, label=name)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{name}_tsne.png"), dpi=120)
    if show:
        plt.show()


@dataclass
class MetaMetrics:
    examples: List[Dict[str, str]] = None

    def to_jsonable_dict(self):
        data_to_dump = {"example": self.examples}
        return data_to_dump

    def __str__(self):
        return "MetaMetrics(For third part extension use case!)"

    def __repr__(self):
        return self.__str__()

@dataclass
class SequencePair:
    first_sequence: str
    second_sequence: str

    def __str__(self):
        return f"sentence1: {self.first_sequence}\t sentence2: {self.second_sequence}"

@dataclass
class SequencePairBatch:
    first_sequence_batch: List[str]
    second_sequence_batch: List[str]

    def __post_init__(self):
        self._batch: List[SequencePair] = [ SequencePair(f, s) for f, s in zip(self.first_sequence_batch, self.second_sequence_batch)]

    def __len__(self):
        return len(self.first_sequence_batch)

    def __getitem__(self, item):
        return self._batch[item]

def max_length_of_sequences(data, base, dataset: Union[Dataset, Iterable[Dataset]] = None, length_of_seq_call_back: Callable = None) -> Optional[int]:
    assert isinstance(base, int)
    from src.nlps.data import Data
    assert isinstance(data, Data)
    if dataset is None:
        return base

    if isinstance(dataset, Dict):
        lens = np.ndarray([max_length_of_sequences(data, base, d, length_of_seq_call_back) for d in dataset.values()])
        return int(lens.max())
    elif isinstance(dataset, list) or isinstance(dataset, tuple):
        lens = np.ndarray([max_length_of_sequences(data, base, d, length_of_seq_call_back) for d in dataset])
        return int(lens.max())
    else:
        assert isinstance(dataset, Dataset)

    max_length = base

    for sentences in data.extract_input_label_from_samples(dataset):
        for s in sentences:
            length_call = length_of_seq_call_back if isinstance(length_of_seq_call_back, Callable) else lambda s: len(s.split())
            length = length_call(s)
            if length > max_length:
                max_length = length

    return max_length





def update_instant_attribute(source, target):
    t_as = vars(target)
    for s_a in vars(source):
        if s_a in t_as:
            setattr(target, s_a, getattr(source, s_a))


def map_item_wise(map_fun, container, item_identify):
    def _map(item):
        if item_identify(item):
            return map_fun(item)

        if isinstance(item, Iterable):
            result = []
            for i in item:
                result.append(_map(i))
            result = type(item)(result)
            return result
        else:
            raise ValueError

    return _map(container)


@singleton
class MetricComputerProxy:
    def __init__(self):
        import datasets
        self._data2computer: Dict[str, datasets.Metric] = dict()

    def metric_computer(self, name: str, metric_factory: Callable = None):
        if name not in self._data2computer:
            self._data2computer[name] = metric_factory()
        return self._data2computer.get(name)
