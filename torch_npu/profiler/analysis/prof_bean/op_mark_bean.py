import struct
from enum import Enum

from ..profiler_config import ProfilerConfig
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
        self._ts = None
        self._dur = None

    @property
    def pid(self) -> int:
        return int(self._constant_data[OpMarkEnum.PROCESS_ID.value])

    @property
    def tid(self) -> int:
        return int(self._constant_data[OpMarkEnum.THREAD_ID.value])

    @property
    def time_ns(self) -> int:
        opmark_ns = ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[OpMarkEnum.TIME_NS.value])
        return ProfilerConfig().get_local_time(opmark_ns)

    @property
    def corr_id(self) -> int:
        return int(self._constant_data[OpMarkEnum.CORRELATION_ID.value])

    @property
    def origin_name(self) -> str:
        return self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.NAME), "")

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

    @property
    def is_dequeue(self) -> bool:
        return self.is_dequeue_start or self.is_dequeue_end

    @property
    def is_enqueue(self) -> bool:
        return self.is_enqueue_start or self.is_enqueue_end

    @property
    def is_torch_op(self) -> bool:
        return False

    @property
    def ts(self) -> int:
        return self._ts

    @property
    def dur(self) -> int:
        return self._dur

    @ts.setter
    def ts(self, ts: int):
        self._ts = ts

    @dur.setter
    def dur(self, dur: int):
        self._dur = dur
