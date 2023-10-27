from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Any

from ..pool import Pool, UnitToken, Unit


class Buffer(Pool):
    def __init__(self):
        super(Buffer, self).__init__()
        self._token_list:List[UnitToken] = list

    def pop(self, token: UnitToken = None) -> Unit:
        if token is not None:
            return super(Buffer, self).pop(token)

        if len(self._token_list) <= 0:
            return None
        return super(Buffer, self).pop(self._token_list[0])

    def push(self, content) -> UnitToken:
        ut = super(Buffer, self).push(content)
        self._token_list.append(ut)


class InPort(ABC):
    @abstractmethod
    def receive_obj(self, obj: object):
        ...

    def __init__(self):
        self._receive_buffer: Buffer = Buffer()


class OutPort(ABC):
    @abstractmethod
    def send_to(self, next_port: InPort):
        ...

    def __init__(self):
        self._send_buffer: Buffer = Buffer()


class UniversalInPort(InPort):
    def receive_obj(self, obj: object):
        self.receive_buffer.push(obj)

    @property
    def receive_buffer(self):
        return self._receive_buffer

    def __init__(self):
        super(UniversalInPort, self).__init__()


class UniversalOutPort(OutPort):
    def send_to(self, next_port: InPort):
        content = self.send_buffer.pop()
        assert content is not None
        next_port.receive_obj(content)

    @property
    def send_buffer(self):
        return self._send_buffer

    def __init__(self):
        super(UniversalOutPort, self).__init__()


class UniversalPort:

    def receive_obj(self, obj: object):
        pass

    def send_to(self, next_port: InPort):
        pass

    def __init__(self):
        self._in_port:UniversalInPort = UniversalInPort()
        self._out_port: UniversalOutPort = UniversalOutPort()






