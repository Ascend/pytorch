import struct
from enum import Enum
from ..profiler_config import ProfilerConfig
from ..prof_common_func.constant import Constant

__all__ = []


class PythonFuncCallEnum(Enum):
    START_NS = 0
    THREAD_ID = 1
    PROCESS_ID = 2
    TRACE_TAG = 3


class PythonFuncCallBean:
    CONSTANT_STRUCT = "<3QB"
    TLV_TYPE_DICT = {
        Constant.NAME: 2
    }

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))
        self._start_ns = ProfilerConfig().get_local_time(
            ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[PythonFuncCallEnum.START_NS.value]))
        self._tid = int(self._constant_data[PythonFuncCallEnum.THREAD_ID.value])
        self._pid = int(self._constant_data[PythonFuncCallEnum.PROCESS_ID.value])
        self._trace_tag = int(self._constant_data[PythonFuncCallEnum.TRACE_TAG.value])
        self._name = self._origin_data.get(self.TLV_TYPE_DICT.get(Constant.NAME), "")

    @property
    def start_ns(self) -> int:
        return self._start_ns

    @property
    def tid(self) -> int:
        return self._tid

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def trace_tag(self) -> int:
        return self._trace_tag

    @property
    def name(self) -> str:
        return self._name
