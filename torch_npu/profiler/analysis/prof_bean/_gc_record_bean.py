import struct
from enum import Enum

from .._profiler_config import ProfilerConfig
from ..prof_common_func._constant import Constant

__all__ = []


class GCRecordEnum(Enum):
    PID = 0
    STRAT_NS = 1
    END_NS = 2


class GCRecordBean:

    def __init__(self, data):
        self._constant_data = struct.unpack(Constant.GC_RECORD_FORMAT, data)
        self._start_ns = ProfilerConfig().get_local_time(
            ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[GCRecordEnum.STRAT_NS.value]))
        self._end_ns = ProfilerConfig().get_local_time(
            ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[GCRecordEnum.END_NS.value]))
        self._pid = int(self._constant_data[GCRecordEnum.PID.value])
        self._tid = int(self._constant_data[GCRecordEnum.PID.value])

    @property
    def ts(self) -> int:
        return self._start_ns

    @property
    def name(self) -> str:
        return "GC"

    @property
    def pid(self) -> int:
        return self._pid

    @pid.setter
    def pid(self, pid: int):
        self._pid = pid

    @property
    def tid(self) -> int:
        return self._tid

    @property
    def args(self) -> dict:
        return {}

    @property
    def dur(self) -> int:
        return self._end_ns - self._start_ns
