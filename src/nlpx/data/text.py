from abc import ABC

from .data import Data, Dataset, DatasetSplitType
from ..utils.utils import max_length_of_sequences


class TextData(Data, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_length = None
        self._min_length = None

    def _load_dataset(self, split: DatasetSplitType, *args, **kwargs) -> Dataset:
        result = super()._load_dataset(split)
        self._update_max_length(result)
        self._update_min_length(result)
        return result

    @property
    def max_length(self):
        if not (isinstance(self._max_length, int) and self._max_length > 0):
            return None

        if isinstance(self._min_length, int):
            assert self._min_length <= self._max_length

        return self._max_length

    def _update_max_length(self, dataset: Dataset = None):
        max_length = self._max_length if isinstance(self._max_length, int) else 3
        self._max_length = max_length_of_sequences(data=self, base=max_length, dataset=dataset)

    @property
    def min_length(self):
        if not (isinstance(self._min_length, int) and self._min_length > 0):
            return None

        if isinstance(self._max_length, int):
            assert self._max_length >= self._min_length

        return self._min_length

    def _update_min_length(self, dataset: Dataset = None):
        _min_length = self._min_length if isinstance(self._min_length, int) else 3
        if _min_length > 0:
            _min_length = -1 * _min_length
        _min_length = min(max_length_of_sequences(data=self, base=_min_length, dataset=dataset,
                                                   length_of_seq_call_back=lambda s: -len(s.split())), -2)
        assert _min_length <= 0
        self._min_length = -1 * _min_length
