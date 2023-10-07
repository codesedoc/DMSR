import json
import os.path
import sys

import torch

from experiment import Name2ApproachClass, ArgumentPool, Name2DataClass
from src.nlps.utils.runtime import RunTime


def search_arg(name) -> bool:
    name = f'--{name}'
    args = sys.argv
    if name in args:
        return True
    return False


def add_sys_argv(name, value=None):
    if search_arg(name):
        return
    name = f'--{name}'
    if value is None:
        sys.argv.append(name)
    else:
        sys.argv.extend((name, value))


def test():
    add_sys_argv('do_eval')
    add_sys_argv('do_train')
    add_sys_argv('do_eval')
    add_sys_argv('dataset', 'glue/stsb')
    add_sys_argv('approach', f'test_classification')
    add_sys_argv('plm_name_or_path', 'bert-base-uncased')
    add_sys_argv('output_dir', f'tmp')
    add_sys_argv('smoke_test')


def set_up(do_test=False) -> None:
    if search_arg('args_file'):
        args_file = sys.argv[-1]
        assert os.path.isfile(args_file)
        with open(args_file) as af:
            args_dict: dict = json.load(af)
        flags = args_dict.get('flags', [])
        sys.argv = sys.argv[:1]
        for name, value in args_dict.items():
            sys.argv.extend([f'--{name}', value])
        sys.argv.extend(map(lambda x: f'--{x}', flags))
    else:
        if do_test or not (search_arg('approach') or search_arg('dataset')):
            test()
        else:
            pass

    add_sys_argv('eval_accumulation_steps', '16')
    add_sys_argv('save_strategy', 'no')
    if torch.backends.mps.is_available():
        add_sys_argv('use_mps_device')


def run():
    set_up()
    runtime = RunTime()
    meta_args = ArgumentPool.meta_arguments
    data_class = Name2DataClass[meta_args.dataset]
    approach_class = Name2ApproachClass[meta_args.approach]
    _data = data_class()
    runtime.register(data=_data)
    _approach = approach_class()
    runtime.register(approach=_approach)

    _approach(_data)


if __name__ == '__main__':
    run()
