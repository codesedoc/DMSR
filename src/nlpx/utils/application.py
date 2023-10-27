
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Tuple, Union, List, Iterable

import numpy as np
from datasets import Dataset

from ..argument import Argument, ArgumentConfigurator, ArgumentPool
from ..data import Data, Name2DataClass

from ..approach import Approach


@dataclass
class ApplicationArgument(Argument):
    app_force_io: bool = field(
        default=False,
        metadata={
            'help': "Whether force to ignore the archive last application result, and archive new result.",
        }
    )

    smoke_test: bool = field(
        default=False,
        metadata={
            'help': "Whether force to ignore the archive last application result, and archive new result.",
        }
    )

    app_data: str = field(
        default=tuple(),
        metadata={
            'help': "Specify the name of dataset(s) for application. Splited by ','.",
        }
    )

    output_dir: str = field(
            default=tuple(),
            metadata={
                'help': "Specify the output_dir for application.",
            }
    )

class Application(ArgumentConfigurator):
    argument_class = ApplicationArgument
    @classmethod
    def collect_argument(cls, *arg, **kwargs):
        ArgumentPool.push(arg_class=cls.argument_class)

    def assign_argument(self, *arg, **kwargs):
        self.args: ApplicationArgument = ArgumentPool.pop(self.argument_class)

    def __init__(self, approach, data:Data = None,  data_name: str = None, output_dir: str = None, **kwargs):
        super(Application, self).__init__(reset_argument=True)
        self._data_list: List[Data] = list()

        def instance_data_from_name(name):
            data_class: type = Name2DataClass.get(name, None)
            if data_class is None:
                raise ValueError(
                    f"\nThe specified name ({name}) of application dataset is invalid or out of range!\n")
            return data_class()

        if data is not None:
            data = [data,]
        elif data_name is not None:
            data = [instance_data_from_name(data_name),]
        elif len(self.args.app_data)>0:
            names = self.args.app_data.split(",")
            if len(names) == 0:
                raise ValueError
            data = [instance_data_from_name(name.strip()) for name in names]
        else:
            raise ValueError
        self._data_list: List[Data] = data
        self.data: data = None
        self.approach: Approach = approach
        self.args.output_dir = output_dir if isinstance(output_dir, str) else approach.args.application_dir
        self.args.smoke_test =  kwargs.get('smoke_test', self.args.smoke_test)
        self.args.app_force_io = kwargs.get('force_io',  self.args.app_force_io)
        if not os.path.exists(self.args.output_dir):
            os.system(f'mkdir -p {self.args.output_dir}')

    def preprocess(self, runtime: Dict[str, Any], dataset_collator: Callable= None):
        if not isinstance(runtime['dataset'], Dataset):
            if not isinstance(dataset_collator, Callable):
                self.data.application_dataset_collate(runtime)
            else:
                dataset_collator(self.data, runtime)

    def process(self, runtime: Dict[str, Any], processor: Callable = None):
        dataset: Dataset = runtime['dataset']
        if dataset is None:
            return None

        if not isinstance(processor, Callable):
            self.approach.application(self.data, runtime)
        else:
            processor(self.approach, self.data, runtime)

    def post_process(self, runtime: Dict[str, Any], processor: Callable = None):
        try:
            if not isinstance(processor, Callable):
                self.data.application_finish_call_back(runtime)
            else:
                processor(self.data, runtime)
        except:
            raise
        finally:
            print(f"Archiving the application result for data {self.data.abbreviation}")
            self.archive_application_result(runtime)

    def _archive_file(self, runtime: Dict[str, Any]):
        output_dir = self.args.output_dir

        if not isinstance(output_dir, str):
            raise ValueError

        if not os.path.exists(output_dir):
            raise ValueError

        if not os.path.exists(self.data.data_dir()):
            self.data.download()

        test = runtime.get('smoke_test', False)

        if test:
            archive_file = os.path.join(output_dir, f'{self.data.abbreviation.replace("/", "-")}_test.pk')
        else:
            archive_file = os.path.join(output_dir, f'{self.data.abbreviation.replace("/", "-")}.pk')
        return archive_file

    def try_load_application_result(self, runtime):
        force = self.args.app_force_io
        archive_file = self._archive_file(runtime)
        if os.path.isfile(archive_file) and not force:
            with open(archive_file, mode='rb') as f:
                old_runtime = pickle.load(f)
                runtime["dataset"] = old_runtime["dataset"]
                runtime["dataset_split_type"] = old_runtime["dataset_split_type"]
                return True
        return False

    def archive_application_result(self, runtime):
        force = runtime.get('force_io', False)
        archive_file = self._archive_file(runtime)
        if not os.path.isfile(archive_file) or force:
            with open(archive_file, mode='wb') as f:
                pickle.dump(runtime, f)

    def run(self, dataset_collator=None, processor=None, post_processor=None, data:Data=None, **kwargs):
        result = 0
        if data is None:
            for d in self._data_list:
                if not isinstance(d, Data):
                    raise ValueError
                result += self.run(dataset_collator, processor, post_processor, d, **kwargs)
            return result
        self.data = data
        print(f"**** Apply approach {self.approach.abbreviation} to data {data.abbreviation} ****")
        runtime: Dict[str, Any] = {
            'dataset': None,
            'smoke_test': self.args.smoke_test,
            'force_io': self.args.app_force_io,
            'output_dir': self.args.output_dir,
            'dataset_split_type': None
        }
        runtime.update(kwargs)
        if self.try_load_application_result(runtime):
            self.post_process(runtime, post_processor)
            return result
        self.preprocess(runtime, dataset_collator)
        self.process(runtime, processor)
        self.post_process(runtime, post_processor)
        self.data = None
        return result

