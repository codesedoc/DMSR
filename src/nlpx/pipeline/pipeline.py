from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Tuple, Dict, Any
from .port import UniversalPort, UniversalOutPort, UniversalInPort


class _Procedure(ABC):
    @abstractmethod
    def initial(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def finish(self):
        pass


class Node(_Procedure):
    def initial(self, *args, **kwargs):
        raise RuntimeError

    def execute(self):
        raise RuntimeError

    def finish(self):
        raise RuntimeError

    @property
    def in_port(self):
        return  self._in_port

    @property
    def out_port(self):
        return self._out_port

    def __init__(self):
        self._in_port: UniversalInPort = UniversalInPort()
        self._out_port: UniversalOutPort = UniversalOutPort()
        self._next_node: Node = Node
        # self.initial()


def _isextensible(extension: Any):
    return hasattr(extension)


class ExtensibleNode(Node):

    # def initial(self, extension: Any):
    #     super(ExtensibleNode, self).initial()
    #
    # def execute(self):
    #     super(ExtensibleNode, self).execute()
    #     _data = self.in_port.receive_buffer.pop()
    #     self.out_port.send_buffer.push(_data)
    #     self.out_port.send_to(self._next_node)
    #
    # def finish(self):
    #     super(ExtensibleNode, self).finish()

    def __init__(self):
        super(ExtensibleNode, self).__init__()
        self._ex_io_port: UniversalPort = UniversalPort()


class PipelineState(Enum):
    FREE = auto()
    BUSY = auto()


class Pipeline(_Procedure):
    # def initial(self):
    #     self._node_sequence = tuple(eval(f'{nc.__name__}()') for nc in self._node_classes)
    #     for i, n in enumerate(self._node_sequence[:-1]):
    #         n._next_node = self._node_sequence[i+1]
    #
    # def execute(self):
    #     for node in self._node_sequence:
    #         node.execute()
    #
    # def finish(self):
    #     for node in self._node_sequence:
    #         node.finish()

    def initial(self):
        self._node_sequence = tuple(eval(f'{nc.__name__}()') for nc in self._node_classes)
        for i, n in enumerate(self._node_sequence[:-1]):
            n._next_node = self._node_sequence[i+1]

    def execute(self):
        for node in self._node_sequence:
            node.execute()

    def finish(self):
        for node in self._node_sequence:
            node.finish()

    def __len__(self):
        return len(self._node_sequence)

    def __str__(self):
        if len(self._node_classes) == 0:
            return '()'
        return str((self._node_classes[0], *(f'-->{node}' for node in self._node_classes[1:])))

    def __init__(self, node_classes: List[type]):
        self._node_sequence: Tuple[Node] = list()
        self._state: PipelineState
        for nc in node_classes:
            assert issubclass(nc, Node)
        self._node_classes = node_classes
        self._id2node: Dict[int, Node] = dict()
        self.initial()


class SimplePipeline(Pipeline):

    def execute(self):
        from argument import ArgumentPool, ArgumentParser, MetaArgument
        from data import Name2DataClass
        from approach import Name2ApproachClass

        meta_args, = ArgumentParser.fast_parse(MetaArgument)
        data_class = Name2DataClass[meta_args.dataset]
        approach_class = Name2ApproachClass[meta_args.approach]
        data_class.collect_argument()
        approach_class.collect_argument()
        paser = ArgumentParser()
        paser.register_argclass(ArgumentPool.arg_classes)
        for ac, args in zip(ArgumentPool.arg_classes, paser.parse()):
            ArgumentPool.update(ArgumentPool.ArgUnit(arg_class=ac, namespace=args))
        _data = data_class()
        _approach = approach_class()
        _approach(_data)
