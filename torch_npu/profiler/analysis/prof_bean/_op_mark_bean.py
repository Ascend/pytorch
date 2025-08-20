import struct
from enum import Enum

from .._profiler_config import ProfilerConfig
from ..prof_common_func._constant import Constant

__all__ = []


class OpMarkEnum(Enum):
    TIME_NS = 0
    CATEGORY = 1
    CORRELATION_ID = 2
    THREAD_ID = 3
    PROCESS_ID = 4


class _OpMarkCategoryEnum(Enum):
    ENQUEUE_START = 0
    ENQUEUE_END = 1
    DEQUEUE_START = 2
    DEQUEUE_END = 3


class OpMarkBean:
    TLV_TYPE_DICT = {
        Constant.NAME: 1
    }
    CONSTANT_STRUCT = "<q4Q"
    CONSTANT_UNPACKER = struct.Struct(CONSTANT_STRUCT)

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = self.CONSTANT_UNPACKER.unpack(data.get(Constant.CONSTANT_BYTES))
        self._category = _OpMarkCategoryEnum(self._constant_data[OpMarkEnum.CATEGORY.value])
        self._pid = None
        self._tid = None
        self._time_ns = None
        self._corr_id = None
        self._origin_name = None
        self._name = None
        self._args = None

    @property
    def pid(self) -> int:
        if self._pid is None:
            self._pid = self._constant_data[OpMarkEnum.PROCESS_ID.value]
        return self._pid

    @property
    def tid(self) -> int:
        if self._tid is None:
            self._tid = self._constant_data[OpMarkEnum.THREAD_ID.value]
        return self._tid

    @property
    def time_ns(self) -> int:
        if self._time_ns is None:
            self._init_time_ns()
        return self._time_ns

    @property
    def corr_id(self) -> int:
        if self._corr_id is None:
            self._corr_id = self._constant_data[OpMarkEnum.CORRELATION_ID.value]
        return self._corr_id

    @property
    def origin_name(self) -> str:
        if self._origin_name is None:
            self._origin_name = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.NAME), "")
        return self._origin_name

    @property
    def name(self) -> str:
        if self._name is None:
            if self.is_dequeue_start or self.is_dequeue_end:
                self._name = "Dequeue@" + self.origin_name
            else:
                self._name = "Enqueue"
        return self._name

    @property
    def args(self) -> dict:
        if self._args is None:
            self._args = {"correlation_id": self.corr_id}
        return self._args

    @property
    def is_enqueue_start(self) -> bool:
        return self._category == _OpMarkCategoryEnum.ENQUEUE_START

    @property
    def is_enqueue_end(self) -> bool:
        return self._category == _OpMarkCategoryEnum.ENQUEUE_END

    @property
    def is_dequeue_start(self) -> bool:
        return self._category == _OpMarkCategoryEnum.DEQUEUE_START

    @property
    def is_dequeue_end(self) -> bool:
        return self._category == _OpMarkCategoryEnum.DEQUEUE_END

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

    def _init_time_ns(self):
        profiler_config = ProfilerConfig()
        syscnt = self._constant_data[OpMarkEnum.TIME_NS.value]
        self._time_ns = profiler_config.get_local_time(
            profiler_config.get_timestamp_from_syscnt(syscnt))
