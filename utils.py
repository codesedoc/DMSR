from enum import StrEnum
import json
from logging import Logger
from pathlib import Path
import random
from typing import Any, Callable
from nlpe import ArgumentPool
from nlpe.utils import Glossary, normalize_str_arg
import numpy as np
import torch


class GlossaryEnum(StrEnum):
    @property
    def value(self):
        if not isinstance(self._value_, Glossary):
            self._value_ = Glossary(self._value_)
        return super().value
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return str(self)


def get_unified_model_type_str(model_name):
    model_name = normalize_str_arg(model_name)
    return model_name + "-" + ArgumentPool().meta_argument['approach']


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        result = dict()
        for k, v in obj.items():
            result[str(k)] = jsonable(v)
        return result
    elif isinstance(obj, tuple) or isinstance(obj, list):
        result = []
        for v in obj:
            result.append(jsonable(v))
        return type(obj)(result)
    else:
        result = obj
        try:
            json.dumps(obj)
        except:
            result = str(obj)
        return result
    
    
def log_segment(logger: Callable, title: str, content: str):
    assert isinstance(logger, Callable)
    title = normalize_str_arg(title)
    content = normalize_str_arg(content)
    logger('=' * 100)
    logger('-' * 100)
    logger(title)
    logger('-' * 100)
    for line in content.split('\n'):
        logger(line)
    logger('x' * 100)
    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    

# def remove_dir(path_dir: Path):
#     assert isinstance(path_dir, Path)
#     for root, dirs, files in path_dir.walk(top_down = False, on_error=print):
#         for f in files:
#             Path(f).unlink()
#         for d in dirs:
#             Path(d).rmdir()
#     path_dir.rmdir()