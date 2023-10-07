import argparse
import copy
import dataclasses
import os
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields, make_dataclass, Field, field
from enum import Enum, auto
from typing import Dict, Any, Tuple, Union, Callable, Type, List, Optional, Iterable

import numpy as np

from ..pool import Pool, Unit, UnitToken
from ..utils import class_name_chain
from ..utils.design_patterns import singleton
from ..utils.utils import jsonable


@singleton
class _NECESSARY:
    pass


NECESSARY = _NECESSARY()


class _NULL_ARGUMENT:
    pass


NULL_ARGUMENT = _NULL_ARGUMENT()

def _function():
    pass

FUNCTION_TYPE = type(_function)


class ArgumentType (Enum):
    DATA = auto()
    APPROACH = auto()
    GENERAL = auto()
    EXTERNAL = auto()
    META = auto()


class ArgumentValueError(ValueError):
    ...


@dataclass
class Argument:
    @property
    def atype(self):
        return ArgumentType.GENERAL

    @property
    def name(self):
        return self._name

    identification = uuid.uuid4()

    @property
    def necessary_argument(self):
        result: Dict[str, Any] = {}
        for f in fields(self):
            if f.default is NECESSARY:
                result[f.name] = getattr(self, f.name)
        return result

    @staticmethod
    def is_required_field(field_: Union[Field, Iterable[Field]]):
        if isinstance(field_, Field):
            result = False
            if field_.default is dataclasses.MISSING:
                result = True
            if field_.default is NECESSARY:
                result = True

        elif isinstance(field_, Iterable):
            result = [Argument.is_required_field(f) for f in field_]
        else:
            result = None

        return result

    @classmethod
    def instance(cls):
        args: Dict[str, Any] = {}
        for f in fields(cls):
            args[f.name] = eval(f'{f.type.__name__}()')
        return cls(**args)

    @staticmethod
    def update_defaults_for_fields(cls, field_name2default: Dict[str, Any]):
        parent_class = cls.__bases__
        _fields = fields(cls)
        for f in _fields:
            if f.name in field_name2default:
                f.default = field_name2default[f.name]
        if not issubclass(cls, Argument):
            for f in _fields:
                if f.name in field_name2default:
                    setattr(cls, f.name, f)
            return cls
        else:
            _cls = make_dataclass(cls_name= cls.__name__, fields=[(x.name, x.type, x) for x in _fields], bases=(parent_class,))

            if ArgumentPool.search(cls):
                ArgumentPool.update(arg_unit=ArgumentPool.ArgUnit(arg_class=cls))
            return _cls

    @staticmethod
    def has_metadata(cls, metadata_name):
        _fields = fields(cls)
        for f in _fields:
            if f.metadata.get(metadata_name, None):
                return True
        return False

    @staticmethod
    def have_conflict_fields(cls):
        return Argument.has_metadata(cls, 'conflict')

    @staticmethod
    def is_conflict_field(_field):
        return _field.metadata.get('conflict', None)

    @staticmethod
    def set_conflict_fields(_fields):
        for f in _fields:
            f.metadata = dict(f.metadata)
            f.metadata['conflict'] = True

    @staticmethod
    def is_invalid(arg_class):
        return not issubclass(arg_class, Argument) and arg_class.atype is not ArgumentType.EXTERNAL

    @staticmethod
    def revise_fields(cls, _fields: Union[Field, List[Field], Tuple[Field]]):
        f_name = set()
        if isinstance(_fields, Field):
            _fields = _fields,

        Argument.set_conflict_fields(_fields)

        for f in _fields:
            f_name.add(f.name)
            f.init = False
            # f.default_factory = lambda: ArgumentPool[copy.deepcopy(f.name)]
            f.default_factory = lambda: None

        _cls = make_dataclass(cls_name=cls.__name__, fields=[(x.name, x.type, x) for x in _fields],
                              bases=(cls,))
        cls.__init__ = _cls.__init__

        return cls

    @staticmethod
    def fields(cls) -> Tuple[Field]:
        return fields(cls)

    @staticmethod
    def field_name2value(instance) -> Dict[str, Any]:
        if not isinstance(instance, Argument):
            if instance.atype != ArgumentType.EXTERNAL:
                raise ValueError
        arg_class = instance.__class__
        _fields = Argument.fields(arg_class)
        result = {f.name: getattr(instance, f.name) for f in _fields}
        result['class_name_chain'] = class_name_chain(arg_class)
        return result

    @staticmethod
    def field_name2value_jsonable(instance) -> Dict[str, Any]:
        result = jsonable(Argument.field_name2value(instance))
        return result

    def _check_argument(self) -> None:
        for name, value in self.necessary_argument.items():
            if getattr(self, name) is NECESSARY:
                raise ArgumentValueError(f"{self.__class__}'s field ({name}): need be assigned a value with type "
                                         f"{self.__class__.__dict__.get('__annotations__')[name]}, " )
                                         # f"but a value with type {type(getattr(self, name))} is gaven!")

    def __post_init__(self) -> None:
        self._name: str = self.__class__.__name__
        self._check_argument()


    @staticmethod
    def copy_args(source_arg, target_arg):
        for name, value in vars(source_arg).items():
            setattr(target_arg, name, value)


def argument_class(cls: type = None, bases: Union[type, Tuple[type, ...], _NULL_ARGUMENT] = Argument,
                   force: bool = False, **kwargs) -> Union[Callable]:
    if isinstance(bases, type):
        bases = (bases,)

    def wrap(_cls: type, *args):
        if isinstance(bases, _NULL_ARGUMENT):
            _cls.atype = ArgumentType.EXTERNAL
            return _cls

        _cls = dataclass(_cls)
        if issubclass(_cls, Argument) and not force:
            return _cls

        preserve = {name: getattr(_cls, name) for name in dir(_cls) if isinstance(getattr(_cls, name), property)}

        module_name = _cls.__module__
        _cls_str = str(_cls)
        class_name = _cls_str[9 + len(module_name):-2]

        from types import FunctionType
        callable = {k: v for k, v in vars(_cls).items() if isinstance(v, FunctionType) and v.__qualname__.startswith(class_name)}
        if callable:
            preserve.update(callable)
        # preserve['__post_init__']
        # noinspection PyUnresolvedReferences

        # noinspection PyDataclass
        class_name_prefix = kwargs.get('class_name_prefix', '')
        if class_name_prefix == '':
            if not (_cls_str.find(module_name) and _cls_str.startswith("<class '") and _cls_str.endswith("'>")):
                raise RuntimeError
            class_name_prefix = class_name[:-len(_cls.__name__)]

        # noinspection PyDataclass
        _fields = fields(_cls)
        for f in _fields:
            if f.default is dataclasses.MISSING:
                f.default = NECESSARY

        __cls__ = make_dataclass(cls_name= class_name_prefix + _cls.__name__, fields=[(x.name, x.type, x) for x in _fields], bases=bases)
        _cls.__init__ = __cls__.__init__
        # for name in preserve.keys():
        #     exec(f'__cls__.{name} = preserve["{name}"]', )
        # __cls__.__pre_module__ = module_name
        # __cls__.__pre_str__ = _cls_str
        return _cls
    if cls is None:
        return wrap

    return wrap(cls)


class PATH_MODE(Enum):
    ABSTRACT = 'abs'
    RELATED = 'rel'

    def __str__(self):
        return self.value


@argument_class
class MetaArgument(Argument):
    @property
    def atype(self):
        return ArgumentType.META

    dataset: str = field(
        default=NECESSARY,
        metadata={
            'help': "Specify the dataset need to be deal with.",
        }
    )
    approach: str = field(
        default=None,
        metadata={
            'help': "Specify the approach",
        }
    )
    path_mode: Optional[str] = field(
        default='abs',
        metadata={
            'help': "Specify the mode of path.",
            'choices': "['abs', 'rel']"
        }
    )
    cache_dir: str = field(
        default='./.cache',
        metadata={
            'help': "Specify the dir of cached files, the default is './.cache'"
        }
    )

    force_cache: bool = field(
        default=False,
        metadata={'help': "Whether or not force to download cached info, such as configuration, "
                          "when cached file is exist."})

    force_download: bool = field(
        default=False,
        metadata={'help': "Whether or not force to download online info, such as dataset, configuration, "
                          "when local file is exist."})

    @property
    def _force_flag_names(self):
        return 'force_cache',

    def __post_init__(self) -> None:
        super(MetaArgument, self).__post_init__()
        self.path_mode = PATH_MODE(self.path_mode)
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        if self.force_download:
            for n in self._force_flag_names:
                setattr(self, n, True)

    # model: str = NECESSARY


@argument_class
class ExternalArgument(Argument):
    @property
    def atype(self):
        return ArgumentType.EXTERNAL


@argument_class
class DataArgument(Argument):
    @property
    def atype(self):
        return ArgumentType.DATA

    data_root_dir: str = field(default='storage/dataset',
                               metadata={'help': "The root path of data which save the raw and preprocessed "
                                                 "files of current data"})

    application_dataset: str = field(default=None,
                                     metadata={'help': "Specify the name of dataset for application"})

    force_preprocess: bool = field(default=False,
                                   metadata={'help': "Whether or not preprocess the raw data forcibly"})

    smoke_test: bool = field(
        default=False,
        metadata={
            'help': "Whether conducting smoke test."
        }
    )

    def __post_init__(self) -> None:
        meta_arguments = ArgumentPool.meta_arguments
        if meta_arguments.path_mode is PATH_MODE.ABSTRACT:
            self.data_root_dir = os.path.join(os.getcwd(), self.data_root_dir)


@argument_class
class ApproachArgument(Argument):
    @property
    def atype(self):
        return ArgumentType.APPROACH

    @property
    def application_dir(self):
        application_dir = os.path.join(self.output_dir, 'application')
        return application_dir

    dataset: str = field(
        default=NECESSARY,
        metadata={
            'help': "Specify the dataset need to be deal with.",
        }
    )

    output_dir: str = field(
        default='output',
        metadata={
            'help': "Specify the output dir for processing data."
        }
    )

    tune_hp: bool = field(
        default=False,
        metadata={
            'help': "Whether tuning hyperparameter."
        }
    )

    num_trials: int = field(
        default=128,
        metadata={
            'help': "The number of trials when tuning hyperparameter."
        }
    )

    smoke_test: bool = field(
        default=False,
        metadata={
            'help': "Whether conducting smoke test."
        }
    )

    do_apply: bool = field(
        default=False,
        metadata={
            'help': "Whether conduct the application to target dataset"
        }
    )


@argument_class
class ModelArgument(ExternalArgument):
    def __post_init__(self):
        super(ModelArgument, self).__post_init__()
        self._model_name = None

    @property
    def model_name(self) -> str:
        return self._model_name


class FieldConflictError(ValueError):
    pass


@singleton
class _ArgumentPool(Pool):
    class ArgUnit:
        def __init__(self, arg_class: type, argument: Any = None):
            self._arg_class = arg_class
            self._argument = argument

        @property
        def class_name(self):
            return str(self._arg_class)

        @property
        def arg_class(self):
            return self._arg_class

        @property
        def argument(self) -> Any:
            return self._argument

    def __init__(self):
        super().__init__()
        self._class_name2unit_token: [str, UnitToken] = dict()
        self._all_args: Argument = Argument()
        self._meta_arguments: MetaArgument = None

    # def __copy__(self):
    #     result = self.__class__()
    #     result._all_args = self._all_args
    #     breakpoint()
    #     return result

    def _get_token(self, class_name: str):
        token = self._class_name2unit_token.get(class_name, None)
        # if token is None:
        #     raise ValueError
        return token

    def _set_attr_to_all_args(self, argument: Any):
        if argument is not None:
            for f in fields(argument):
                if f.metadata.get('conflict_fields', None):
                    continue
                self._all_args.__setattr__(f.name, getattr(argument, f.name))

    def _del_attr_from_all_args(self, argument: Any):
        if argument is not None:
            for f in fields(argument):
                if hasattr(self._all_args, f.name):
                    self._all_args.__delattr__(f.name)

    def _conflict_fields(self, arg_class):
        result = []
        field_set = set()
        for ac_name in self._class_name2unit_token.keys():
            ac = self.search(ac_name).arg_class
            for f in Argument.fields(ac):
                field_set.add(f.name)
        for f in Argument.fields(arg_class):
            if f.name in field_set or Argument.is_conflict_field(f):
                result.append(f)

        return result

    def push(self, arg_class: type, _argument: Any = None, **kwargs):
        if self.search(arg_class):
            raise ValueError
        c_f = self._conflict_fields(arg_class)

        if c_f:
            if isinstance(arg_class, Argument):
                raise ValueError
            revise_cls = kwargs.get('revise_cls', None)
            if revise_cls:
                arg_class = Argument.revise_fields(arg_class, c_f)
            else:
                raise FieldConflictError

        arg_unit = self.ArgUnit(arg_class=arg_class, argument=_argument)
        _token = super().push(Unit(name=arg_unit.class_name, content=arg_unit))
        self._set_attr_to_all_args(arg_unit.argument)
        self._class_name2unit_token[arg_unit.class_name] = _token

        return arg_class

    def pop(self, arg_class: type, *args, **kwargs) -> ArgUnit:
        class_name = str(arg_class)
        arg_unit = super().pop(token=self._class_name2unit_token.pop(class_name)).content
        # self._del_attr_from_all_args(arg_unit.argument)
        return arg_unit.argument

    def update(self, arg_class, argument, *args, **kwargs) -> bool:
        arg_unit = ArgumentPool.ArgUnit(arg_class=arg_class, argument=argument)
        if arg_unit.argument is None:
            self._del_attr_from_all_args(self.search(arg_unit.class_name).argument)
        else:
            self._set_attr_to_all_args(arg_unit.argument)
        super().update(token=self._get_token(arg_unit.class_name), content=arg_unit)

    def search(self, arg_class: type, *args, **kwargs) -> Union[None, ArgUnit]:
        class_name = str(arg_class)
        token = self._get_token(class_name)
        if token is None:
            return None
        return super().search(token=token).content

    def __getitem__(self, item):
        if hasattr(self._all_args, item):
            return getattr(self._all_args, item)
        return NECESSARY

    @property
    def meta_arguments(self) -> MetaArgument:
        if self._meta_arguments is None:
            from ..argument import ArgumentParser
            self._meta_arguments, = ArgumentParser().fast_parse(arg_class=MetaArgument)
        return self._meta_arguments

    @meta_arguments.setter
    def meta_arguments(self, value):
        if isinstance(value, MetaArgument):
            self._meta_arguments = value

    @property
    def arg_classes(self) -> Tuple[type]:
        return (unit.content._arg_class for unit in self._unit_table.values())


ArgumentPool: _ArgumentPool = _ArgumentPool()

