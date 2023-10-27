import argparse
import copy
import sys
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Dict, Tuple, Union, Iterable, Set, List, Any
from transformers import HfArgumentParser
from .argument import Argument, DataArgument, ArgumentType, ArgumentPool, MetaArgument
from ..utils.design_patterns import singleton


class RegisterTable:
    class BackBoneNode:
        def __init__(self, branch):
            self.branch: RegisterTable.BranchNode = None
            self.last_branch_node: RegisterTable.BranchNode = None
            self.pre_node: RegisterTable.BackBoneNode = None
            self.next_node: RegisterTable.BackBoneNode = None

    class BranchNode:
        def __init__(self, name, content):
            self.pre_node: RegisterTable.BranchNode = None
            self.next_node: RegisterTable.BranchNode = None
            self.name = name
            self.content = content


    def __init__(self):
        self._last_bone_node: RegisterTable.BackBoneNode = None
        self._back_bone: RegisterTable.BackBoneNode = None

        pass

    def insert(self, cls_name, arg_cls, individual=False):
        if Argument.is_invalid(arg_cls):
            raise ValueError
        node = self.BranchNode(cls_name, arg_cls)
        if self._back_bone is None:
            bone_node = self.BackBoneNode(node)
            self._back_bone = bone_node
            self._last_bone_node = bone_node

        elif individual:
            bone_node = self.BackBoneNode(node)
            self._last_bone_node.next_node = bone_node
            bone_node.pre_node = self._last_bone_node.pre_node
            self._last_bone_node = bone_node
        else:
            bone_node = self._last_bone_node

        if bone_node.branch is None:
            bone_node.branch = node
            bone_node.last_branch_node = node
        else:
            bone_node.last_branch_node.next_node = node
            node.pre_node = bone_node.last_branch_node
            bone_node.last_branch_node = node

    def search(self, cls_name):
        if self._back_bone is None:
            return None

        result = None
        bone_node = self._back_bone
        while bone_node:
            branch_node = bone_node.branch
            while branch_node:
                if branch_node.name == cls_name:
                    result = branch_node
                    break
                branch_node = branch_node.next_node
            if result:
                break
            bone_node = bone_node.next_node
        if result:
            setattr(result, 'bone_node', bone_node)
        return result

    def pop(self, cls_name):
        if self._back_bone is None:
            raise ValueError
        node = self.search(cls_name)
        result = node.content
        current_bone_node: RegisterTable.BackBoneNode = node.bone_node
        if node is None:
            raise ValueError
        if node.pre_node is None:
            if current_bone_node is self._last_bone_node:
                self._last_bone_node = current_bone_node.pre_node
            current_bone_node.pre_node.next_node = current_bone_node.next_node
            current_bone_node.next_node.pre_node = current_bone_node.pre_node
        else:
            if node is current_bone_node.last_branch_node:
                current_bone_node.last_branch_node = node.pre_node
            node.pre_node.next_node = node.next_node
            node.next_node.pre_node = node.pre_node

        return result

    def __setitem__(self, cls_name, arg_cls):
        if Argument.have_conflict_fields(arg_cls):
            individual = True
        else:
            individual = False
        self.insert(cls_name, arg_cls, individual)

    def __getitem__(self, cls_name):
        return self.search(cls_name)

    def groups(self):
        result = []
        bone_node = self._back_bone
        while bone_node:
            g = []
            branch_node = bone_node.branch
            while branch_node:
                g.append(branch_node.content)
                branch_node = branch_node.next_node
            if g:
                result.append(g)
            bone_node = bone_node.next_node

        return result


@singleton
class ArgumentParser:

    @property
    def register_table(self):
        return self._register_table

    def __init__(self):
        super().__init__()
        self._register_table: RegisterTable = RegisterTable()
        self._registered_field_name: Set[str] =set()

    def reset(self):
        self.__init__()

    def register_argclass(self, arg_class: Union[type, Iterable[type]]) -> None:
        if isinstance(arg_class, Iterable):
            for ac in arg_class:
                self.register_argclass(ac)

        elif isinstance(arg_class, type):
            if self._register_table.search(arg_class.__name__):
                print(f'{self._register_table}： {arg_class.__name__} has been registered!')
                return None
            if Argument.is_invalid(arg_class):
                raise TypeError
            for f in fields(arg_class):
                if f.name in self._registered_field_name and f.init:
                    raise ValueError
                else:
                    self._registered_field_name.add(f.name)
            self._register_table[arg_class.__name__] = arg_class
        else:
            raise TypeError

    def remove_argclass(self, arg_class: Union[type, Iterable[type]]) -> None:
        if issubclass(arg_class, Iterable):
            for ac in arg_class:
                self.remove_argclass(ac)
        if not issubclass(arg_class, Argument):
            raise TypeError
        arg_class = self._register_table.pop(arg_class.__name__)
        for f in arg_class.fields():
            if f.name in self._registered_field_name:
                self._registered_field_name.pop(f.name)

    def parse(self, *args, **kwargs) -> Tuple[argparse.Namespace]:
        self.register_argclass(ArgumentPool.arg_classes)
        arg_class_groups = self._register_table.groups()

        def parser_classes(_class):
            arguments = self.fast_parse(_class)
            for ac, args in zip(_class, arguments):
                ArgumentPool.update(arg_class=ac, argument=args)
        for g in arg_class_groups:
            parser_classes(g)
        self.reset()
        return None

    def primary_parse(self, arg_class: Union[type, Iterable[type]], return_remaining_strings: bool = False):
        required_args = sys.argv[1:]
        required_fields = []
        if isinstance(arg_class, type):
            fields_ = Argument.fields(arg_class)
            is_required = Argument.is_required_field(fields_)
            for f, i in zip(fields_, is_required):
                if i and f"--{f.name}" not in required_args:
                    required_fields.append(f)
            if len(required_fields) > 0:
                arg_class = Argument.revise_fields(arg_class, required_fields)
            result = self.fast_parse(arg_class, return_remaining_strings)[0]
        else:
            result = []
            for ac in arg_class:
                result.append(self.primary_parse(ac, return_remaining_strings))
        return result

    def _add_sys_argv(self, name, value=None):
        name = f'--{name}'
        args = sys.argv
        if name in args:
            return
        if value is None:
            sys.argv.append(name)
        else:
            sys.argv.extend((name, value))

    @staticmethod
    def fast_parse(arg_class: Union[type, Iterable[type]], return_remaining_strings: bool = False):
        hf_parser = HfArgumentParser(arg_class)
        args = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)
        if return_remaining_strings:
            result = args
        else:
            result = args[:-1]

        return result


class ArgumentConfigurator(ABC):
    def __init__(self, argument=None, reset_argument=False, **kwargs):
        self.args = argument
        if self.args is None and reset_argument:
            self._reset_argument()

    def _reset_argument(self):
        self.collect_argument()
        ArgumentParser().parse()
        self.assign_argument()

    @classmethod
    @abstractmethod
    def collect_argument(cls, *arg, **kwargs):
        pass

    @abstractmethod
    def assign_argument(self, *arg, **kwargs):
        pass