
from typing import Optional

from .design_patterns import singleton
from ..data import Data
from ..approach import Approach


@singleton
class RunTime:
    def __init__(self):
        self._data: Data = None
        self._approach: Approach = None

    @property
    def data(self) -> Optional[Data]:
        return self._data

    @data.setter
    def data(self, value: Data):
        if isinstance(self._data, Data):
            raise RunTime
        self._data = value

    @property
    def approach(self) -> Optional[Approach]:
        return self._approach

    @approach.setter
    def approach(self, value: Approach):
        if isinstance(self._approach, Approach):
            raise RunTime
        self._approach = value

    def register(self, data: Data = None, approach: Approach = None):
        if isinstance(data, Data):
            self.data = data
        if isinstance(approach, Approach):
            self.approach = approach

    def reset(self):
        self._data = None
        self._approach = None

