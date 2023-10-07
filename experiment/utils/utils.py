import csv
import json
import os.path
from abc import ABC, abstractproperty, abstractclassmethod, abstractstaticmethod
from enum import Enum
from typing import Dict, Any, Callable

import torch
from ray import tune

from src.nlps.approach import Approach, Transformer

Name2Setting: Dict[str, type] = dict()

def setting_register(cls):
    if cls.abbreviation not in Name2Setting:
        Name2Setting[cls.abbreviation] = cls
    else:
        raise ValueError
    return cls


class Setting(ABC):
    @classmethod
    @abstractproperty
    def abbreviation(cls):
        ...

    @abstractclassmethod
    def preprocess(self, samples: Dict[str, Any], *args, **kwargs):
        ...

    @abstractclassmethod
    def application(cls, inputs, *args, **kwargs):
        ...

    @abstractclassmethod
    def prepare_raw_input(cls, samples: Dict[str, Any],  **kwargs):
        pass

    @abstractstaticmethod
    def prepare_raw_label(samples: Dict[str, Any],  **kwargs):
        pass


def _update_hp_prepare(preparation, update):
    for name, value in update.items():
        preparation[name] = preparation[name] if isinstance(preparation.get(name), dict) else dict()
        if isinstance(value, dict):
            preparation[name].update(value)
        else:
            preparation[name] = value


def tuning_hp_prepare_stpg(approach: Transformer, preparation=None):
    target_metric = approach.target_metric_for_tuning_hp()
    result = preparation if isinstance(preparation, dict) else dict()
    update = {
        'hp_space' : {
            "per_device_train_batch_size": 2 if approach.args.smoke_test else tune.choice([8, 16, 32]),
            "per_device_eval_batch_size": 2 if approach.args.smoke_test else approach.training_args.per_device_eval_batch_size,
            "num_train_epochs": tune.choice([2, 3, 4]),
            "max_steps": 1 if approach.args.smoke_test else -1
        },

        'scheduler': tune.schedulers.PopulationBasedTraining(
            time_attr="training_iteration",
            metric=target_metric["name"],
            mode=target_metric["mode"],
            perturbation_interval=1,
            hyperparam_mutations={
                "learning_rate": tune.quniform(1e-5, 1e-4, 1e-5),
                # "per_device_train_batch_size": tune.choice([64]),
            },
        ),

        'parameter_columns': {
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },

    }
    _update_hp_prepare(result, update)
    result['reporter'] = tune.CLIReporter(
        parameter_columns=result['parameter_columns'],
        metric_columns=[*approach.metric_names()['evaluation'], "epoch", "training_iteration"],
    )

    return result


def class_name_chain(cls: type):
    supers = type(cls).mro()
    names = [c.__name__ for c in supers]
    return '-->'.join(names)


def read_lines_from_file(file_path:str, storage_form="t"):
    if not os.path.isfile(file_path):
        raise ValueError
    lines = []
    with open(file_path, f"r{storage_form}") as f:
        for row in f:
            row = row.strip()
            if len(row) > 0:
                lines.append(row)

    return lines


def read_from_csv_file(file_path:str, delimiter=",", quotechar ='|'):
    if not os.path.isfile(file_path):
        raise ValueError

    with open(file_path, f"rt") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for row in reader:
            # print(', '.join(row))
            lines.append(row)
    return lines


def dump_info_to_json(information: Any, file_path:str, **kwargs):
    file_dir = os.path.split(file_path)[0]
    if not os.path.isdir(file_dir):
        raise ValueError
    with open(file_path, mode="w") as f:
        json.dump(information, f,**kwargs)


def update_instant_attribute(source, target):
    t_as = vars(target)
    for s_a in vars(source):
        if s_a in t_as:
            setattr(target, s_a, getattr(source, s_a))



