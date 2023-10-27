import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict
from uuid import uuid4 as random_uuid, UUID


class PoolState(Enum):
    FREE = auto()
    BUSY = auto()
    EMPTY = auto()
    FULL = auto()


@dataclass
class Unit:
    name: str
    content: Any
    uuid: UUID = None

    def __hash__(self):
        return hash(self.uuid)

    def __post_init__(self):
        if self.uuid is None:
            self.uuid = random_uuid()


@dataclass
class UnitToken:
    unit_uuid: UUID
    uuid: UUID = None

    def __hash__(self):
        return hash(self.uuid)

    def __post_init__(self):
        if self.uuid is None:
            self.uuid = random_uuid()

POOL_DEFAULT_CAPACITY = 128


class Pool:
    def __init__(self):
        self._state: PoolState = PoolState.FREE
        self._capacity: int = POOL_DEFAULT_CAPACITY
        self._unit_table: Dict[UUID, Unit] = dict()

    def push(self, unit: Unit) -> UnitToken:
        _token = UnitToken(unit_uuid=unit.uuid)
        self._unit_table[_token] = unit
        return _token

    def pop(self, token: UnitToken, *args, **kwargs) -> Unit:
        return self._unit_table.pop(token)

    def search(self, token: UnitToken, *args, **kwargs) -> Unit:
        return self._unit_table.get(token, None)

    def remove(self, token: UnitToken) -> bool:
        pass

    def update(self, token: UnitToken, content: Any, *args, **kwargs):
        self._unit_table[token].content = content
