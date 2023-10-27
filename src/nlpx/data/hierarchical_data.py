import copy
import dataclasses
import json
import os.path
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from queue import SimpleQueue
from typing import Optional, List, Callable, Tuple, Dict, Union, Iterable, Any, Set

import numpy
from datasets import Dataset
from tqdm import tqdm

from .data import Data, DatasetSplitType, ALL_DATASET_SPLIT
from ..utils.design_patterns import init_argument_decorator


class Node:
    def __init__(self, _id, name, samples: Optional[Dataset] = None, parent = None, children=None):
        self._id: str = _id
        self._name = name
        self._samples: Set[int] = set()
        if samples is not None:
            self.samples = samples
        self._parent: Node = None
        self._children: OrderedDict[str, Node] = None
        self.parent = parent
        if children is None:
            children = []
        self._init_children(children)
        self.meta_info: Dict[str, Any] = dict()

    # def printable_structure(self, current_level: int = -1) -> dict:
    #     result = {
    #         'level': current_level,
    #         'id': self._id,
    #         'name': self._name,
    #         'meta_info': str(self.meta_info),
    #         'children': [c.printable_structure(current_level+1) for c in self.children]
    #     }
    #     return result

    @property
    def id(self) -> str:
        return self._id

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, Node):
            raise ValueError("Only value of 'Node' class is accepted.")
        self._parent = value

    def _init_children(self, children: Iterable):
        self._children = OrderedDict()
        for c in children:
            assert isinstance(c, None)
            self.add_child(c)

    @property
    def children(self) -> Tuple:
        return tuple(self._children.values())

    def add_child(self, node) -> bool:
        if not isinstance(node, Node):
            raise ValueError("The type of given value is not 'Node'. ")
        assert node.id not in self._children
        self._children[node.id] = node
        assert node.parent is None
        node.parent = self
        return True

    def remove_child(self, node) -> bool:
        if not isinstance(node, Node):
            raise ValueError("The type of given value is not 'Node'. ")
        self._children.pop(node.id)
        return True

    @property
    def first_child(self):
        if len(self.children) > 0:
            return self.children[0]
        return None

    @property
    def name(self):
        return self._name

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_leaf(self):
        return self._children is None or len(self._children) == 0

    @property
    def level(self):
        if self._parent is None:
            return 0
        else:
            return self._parent.level + 1

    @property
    def samples(self) -> Dataset:
        if self.is_root:
            if not isinstance(self._samples, set) or len(self._samples) == 0:
                posterity = RootNodeInfo(self).posterity
                ids = set()
                for n in posterity:
                    ids.update(n._samples)
                self._samples = ids

        ids = self._samples
        root = self._search_root()
        id2sample = RootNodeInfo(root).id2sample
        result = []
        for i in ids:
            result.append(id2sample[i])
        result = Dataset.from_list(result)
        return result

    @samples.setter
    def samples(self, value: Dataset):
        if self.is_root:
            return

        if not isinstance(value, Dataset):
            raise ValueError("Only value with 'Dataset' type is accepted.")
        root = self._search_root()
        id2sample = RootNodeInfo(root).id2sample
        _samples = set()
        for s in value:
            id2sample[s["id"]] = s
            _samples.add(s["id"])
            _samples.add(s["id"])
        self._samples = _samples
        root._samples = None

    def _search_root(self):
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
        assert current_node.is_root
        return current_node


class RootNodeInfo:
    def __init__(self, root: Node):
        assert isinstance(root, Node)

        if not hasattr(root, 'info'):
            setattr(root, 'info', {
                "id2sample": dict(),
            })
        self._root = root

    @property
    def id2sample(self):
        return getattr(self._root, 'info')["id2sample"]

    @property
    def root(self):
        return self._root

    @property
    def posterity(self) -> List[Node]:
        def _posterity(node: Node):
            result = list(node.children)
            for c in node.children:
                result.extend(_posterity(c))
            return result
        return _posterity(self._root)


class TraverseMode(Enum):
    DFS = "Deep First Search"
    WFS = "Width First Search"


def _hierarchy_args_decorator(*args, **kwargs) -> Tuple[List, Dict]:
    if len(args) > 0 and isinstance(args[0], HierarchyInfo):
        return [], args[0].to_dict()
    return args, kwargs


# @init_argument_decorator(_hierarchy_args_decorator)
@dataclass
class HierarchyInfo:
    _origin_hierarchy: Optional[None]
    depth: int = -1
    nodes: Union[Tuple[Node], List[Node], OrderedDict[str, Node]] = None
    leaves: Union[Tuple[Node], List[Node]] = None
    depth2nodes: Optional[Tuple[Tuple[Node]]] = None

    def __post_init__(self):
        if isinstance(self._origin_hierarchy, HierarchyInfo):
            self.__init__(**self._origin_hierarchy.to_dict())
        assert self._origin_hierarchy is None or isinstance(self._origin_hierarchy, Hierarchy)

    def _printable_info(self) -> Dict:
        return {
            "data_name": self._origin_hierarchy.data_name,
            "data_size": len(self._origin_hierarchy.samples),
            "depth": self.depth,
            "number_of_nodes": len(self.nodes),
            "number_of_leaves": len(self.leaves),
        }

    @property
    def structure(self) -> Optional[Dict[int, Any]]:
        distribution = dict()

        def recursion(node: Node, current_level: int = -1):
            if current_level not in distribution:
                distribution[current_level] = 0
            distribution[current_level] += 1
            return {
                'level': current_level,
                'id': node.id,
                'name': node.name,
                'meta_info': str(node.meta_info),
                'children': [recursion(c, current_level + 1) for c in node.children]
            }
        if not isinstance(self._origin_hierarchy, Hierarchy) or not isinstance(self._origin_hierarchy.root, Node):
            return None

        structure = recursion(self._origin_hierarchy.root, 0)
        return {
            "distribution": str(list(distribution.items())),
            "body": structure
        }

    def to_json(self, output_file, readable: bool = True):
        result = self._printable_info()
        if not self._origin_hierarchy.is_empty:
            result['structure'] = self.structure
        with open(output_file, mode='w') as f:
            json.dump(result, f, indent=4 if readable else None)

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def root(self) -> Optional[Node]:
        result = None
        if isinstance(self._origin_hierarchy, Hierarchy):
            result = self._origin_hierarchy.root
        return result


class MetaHierarchy(ABC):
    @abstractmethod
    def to_hierarchy(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def info(self) -> HierarchyInfo:
        pass


class Hierarchy:
    def __init__(self, data_name: str, data_split: DatasetSplitType = None, root: Node = None):
        self._root = root
        self._data_name = data_name
        self._data_split = data_split

    @classmethod
    def _from_meta_hierarchy(cls, mete_hierarchy):
        assert isinstance(mete_hierarchy, Hierarchy)
        c_meta = copy.deepcopy(mete_hierarchy)
        instance = cls(data_name=c_meta.data_name, root=c_meta.root)
        return instance

    @property
    def data_name(self):
        return self._data_name

    @property
    def data_split(self):
        return self._data_split

    @data_split.setter
    def data_split(self, value: DatasetSplitType):
        assert isinstance(value, DatasetSplitType)
        self._data_split = value

    @property
    def depth(self) -> int:
        if self.is_empty:
            return -1

        meta_info = self.traverse()
        return meta_info.depth

    @property
    def is_empty(self) -> bool:
        return self._root is None

    @property
    def root(self) -> Optional[Node]:
        return self._root

    @root.setter
    def root(self, value: Node):
        if isinstance(value, Node):
            self._root = value
        elif value is None:
            self.clean()
        else:
            raise ValueError

    def traverse(self, operation: Callable = None, model: TraverseMode = TraverseMode.WFS) -> HierarchyInfo:
        if self.is_empty:
            return None

        def _operation(node: Node):
            nonlocal level2nodes
            nonlocal current_depth


            if node.id in meta_info.nodes:
                raise ValueError

            if current_depth not in level2nodes:
                level2nodes[current_depth] = []
            level2nodes[current_depth].append(node)

            if isinstance(operation, Callable):
                operation(node)

            meta_info.nodes[node.id] = node

            if node.is_leaf:
                meta_info.leaves.append(node)
                if current_depth > meta_info.depth:
                    meta_info.depth = current_depth

        def _dfs(extension_point: Node):
            nonlocal current_depth

            if not isinstance(extension_point, Node):
                raise ValueError

            _operation(extension_point)

            current_depth += 1

            for child in extension_point.children:
                _dfs(child)

            current_depth -= 1

        def _wfs(extension_point: Node):
            nonlocal current_depth

            ep_queue = SimpleQueue()
            ep_queue.put(extension_point)
            level_first_node = extension_point
            while not ep_queue.empty():
                extension_point = ep_queue.get()
                if not isinstance(extension_point, Node):
                    raise ValueError

                if level_first_node is None:
                    level_first_node = extension_point.first_child
                elif level_first_node == extension_point:
                    current_depth += 1
                    level_first_node = extension_point.first_child

                _operation(extension_point)

                for child in extension_point.children:
                    ep_queue.put(child)
            assert level_first_node is None

        level2nodes: Dict[int, list] = dict()
        meta_info = HierarchyInfo(_origin_hierarchy=self, nodes=OrderedDict(), leaves=[])

        current_depth = -1
        if model is TraverseMode.DFS:
            _dfs(self._root)
        elif model is TraverseMode.WFS:
            _wfs(self._root)
        else:
            raise ValueError

        meta_info.nodes = tuple(meta_info.nodes.values())
        meta_info.leaves = tuple(meta_info.leaves)
        meta_info.depth2nodes = tuple([tuple(level2nodes[d]) for d in range(meta_info.depth + 1)])

        return meta_info

    @property
    def info(self) -> HierarchyInfo:
        return self.traverse()

    def search(self, node_id) -> Optional[Node]:
        if node_id is None:
            return None
        result = []
        self.traverse(operation=lambda node: result.append(node) if node.id == node_id else None)
        if len(result) == 0:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            raise ValueError("ID of node is duplicated.")

    def insert(self, node: Node, parent_id = None):
        if isinstance(node, Node):
            raise ValueError("Only node with type of 'Node' is accepted.")
        if parent_id is None:
            if self._root is not None:
                raise ValueError("Only the parent's id of root can be 'None'. ")
            self._root = parent_id

        parent = self.search(parent_id)
        if parent is None:
            raise ValueError("The parent node is not exist.")

        assert parent.add_child(node)

    def pop(self, node: Union[Node, str]):
        if isinstance(node, str):
            node = self.search(node)
        if not isinstance(node, Node):
            return None
        if node is self._root:
            self.clean()
        node.parent.remove_child(node)
        node.parent = None
        return node

    def clean(self):
        self._root = None

    @property
    def samples(self) -> Dataset:
        if self._root is None:
            return None
        result = self.root.samples
        return result

    @property
    def leaves(self) -> List[Node]:
        result = list()
        self.traverse(operation=lambda node: result.append(node) if node.is_leaf else None)
        return result

    def root2node(self, node: Node) -> Tuple[Node]:
        assert isinstance(node, Node)
        if node is self._root:
            return node,
        result = [node]
        while isinstance(node.parent, Node):
            node = node.parent
            result.insert(0, node)
        assert node is self._root
        return tuple(result)


def _meta_hierarchy_init(self, *args, **kwargs):
    super(self.__class__, self).__init__(*args, **kwargs)
    self._info = None


def _meta_hierarchy_info(self) -> HierarchyInfo:
        if not isinstance(self._info, HierarchyInfo):
            self._info = super(self.__class__, self).info
        return self._info


def create_meta_hierarchy_class(base: type):
    assert issubclass(base, Hierarchy)
    name = base.__name__.split('.')
    name[-1] = f'Meta{name[-1]}'
    name = '.'.join(name)
    meta_class = type(name, (base, MetaHierarchy), {
        'base': base,
        "__init__": _meta_hierarchy_init,
        "info": property(_meta_hierarchy_info),
        'to_hierarchy': lambda self: self.base._from_meta_hierarchy(self)
    })

    return meta_class


# class _MetaHierarchy(Hierarchy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._info: HierarchyInfo = None
#
#     @property
#     def info(self) -> HierarchyInfo:
#         if not isinstance(self._info, HierarchyInfo):
#             self._info = super().info
#         return self._info


class HierarchicalData(Data, ABC):
    _meta_hierarchy: Hierarchy = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hierarchy: Dict[DatasetSplitType, Hierarchy] = None

    # @property
    # @abstractmethod
    # def information(self):
    #     pass

    @property
    @abstractmethod
    def label_id_column_name(self) -> str:
        raise RuntimeError

    @abstractmethod
    def label_id2samples(self, dataset: Dataset) -> Dict[str, List]:
        raise RuntimeError

    def _create_hierarchy(self, dataset, delete_empty_node: bool = True) -> Hierarchy:
        hierarchy: Hierarchy = self.meta_hierarchy.to_hierarchy()
        label_id2samples = self.label_id2samples(dataset)
        for l_i, samples in tqdm(label_id2samples.items(), desc="Fill nodes"):
            node = hierarchy.search(l_i)
            assert isinstance(node, Node)
            node.samples = Dataset.from_list(samples)
        if delete_empty_node:
            empty_node = []
            hierarchy.traverse(lambda n: empty_node.append(n) if not isinstance(n.samples, Dataset) or len(n.samples) <= 0 else None)
            for n in empty_node:
                hierarchy.pop(n)
        return hierarchy

    def hierarchy(self, split: Union[None, Tuple[DatasetSplitType], DatasetSplitType] = ALL_DATASET_SPLIT) -> Union[Hierarchy, Tuple[Hierarchy]]:
        if isinstance(split, Iterable):
            result = []
            for s in split:
                result.append(self.hierarchy(s))
            result = type(split)(result)

        elif isinstance(split, DatasetSplitType):
            print(f'Creat hierarchy for {split.value} set')
            if self._hierarchy is None:
                self._hierarchy = dict()
            if split not in self._hierarchy:
                result = self._create_hierarchy(self.dataset(split))
                assert isinstance(result, Hierarchy)
                result.data_split = split
                self._hierarchy[split] = result
            result = self._hierarchy[split]
        elif split is None:
            return None
        else:
            raise ValueError
        return result

    @classmethod
    @abstractmethod
    def _init_meta_hierarchy(cls) -> MetaHierarchy:
        pass

    @classmethod
    @property
    def meta_hierarchy(cls) -> MetaHierarchy:
        if cls._meta_hierarchy is None:
            cls._meta_hierarchy = cls._init_meta_hierarchy()

        assert isinstance(cls._meta_hierarchy, MetaHierarchy)
        return cls._meta_hierarchy

