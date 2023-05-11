import struct
from enum import Enum

from ..prof_common_func.constant import Constant


class OpMarkEnum(Enum):
    TIME_NS = 0
    CATEGORY = 1
    CORRELATION_ID = 2
    THREAD_ID = 3
    PROCESS_ID = 4


class OpMarkBean:
    TLV_TYPE_DICT = {
        Constant.NAME: 2
    }
    CONSTANT_STRUCT = "<q4Q"

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))

    @property
    def pid(self) -> int:
        return int(self._constant_data[OpMarkEnum.PROCESS_ID.value])

    @property
    def tid(self) -> int:
        return int(self._constant_data[OpMarkEnum.THREAD_ID.value])

    @property
    def time_us(self) -> float:
        return int(self._constant_data[OpMarkEnum.TIME_NS.value]) / 1000.0

    @property
    def corr_id(self) -> int:
        return int(self._constant_data[OpMarkEnum.CORRELATION_ID.value])

    @property
    def name(self) -> str:
        if self.is_dequeue_start or self.is_dequeue_end:
            return "Dequeue@" + str(self._origin_data[self.TLV_TYPE_DICT.get(Constant.NAME)])
        return "Enqueue"

    @property
    def args(self) -> dict:
        return {"correlation_id": self.corr_id}

    @property
    def is_enqueue_start(self) -> bool:
        return int(self._constant_data[OpMarkEnum.CATEGORY.value]) == 0

    @property
    def is_enqueue_end(self) -> bool:
        return int(self._constant_data[OpMarkEnum.CATEGORY.value]) == 1

    @property
    def is_dequeue_start(self) -> bool:
        return int(self._constant_data[OpMarkEnum.CATEGORY.value]) == 2

    @property
    def is_dequeue_end(self) -> bool:
        return int(self._constant_data[OpMarkEnum.CATEGORY.value]) == 3
