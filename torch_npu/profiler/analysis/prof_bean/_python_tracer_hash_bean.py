import struct
from enum import Enum
from ..prof_common_func._constant import Constant

__all__ = []


class PythonTracerHashEnum(Enum):
    KEY = 0


class PythonTracerHashBean:

    CONSTANT_STRUCT = "<Q"
    TLV_TYPE_DICT = {
        Constant.VALUE: 2
    }

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))
        self._key = int(self._constant_data[PythonTracerHashEnum.KEY.value])
        self._value = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.VALUE), "")

    @property
    def key(self) -> int:
        return self._key

    @property
    def value(self) -> str:
        return self._value
