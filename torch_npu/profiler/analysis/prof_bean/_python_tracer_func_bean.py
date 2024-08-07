import struct
from enum import Enum
from .._profiler_config import ProfilerConfig

__all__ = []


class PythonTracerFuncEnum(Enum):
    START_NS = 0
    THREAD_ID = 1
    PROCESS_ID = 2
    HASH_KEY = 3
    TRACE_TAG = 4


class PythonTracerFuncBean:

    CONSTANT_STRUCT = "<4QB"

    def __init__(self, data):
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data)
        self._start_ns = ProfilerConfig().get_local_time(
            ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[PythonTracerFuncEnum.START_NS.value]))
        self._tid = int(self._constant_data[PythonTracerFuncEnum.THREAD_ID.value])
        self._pid = int(self._constant_data[PythonTracerFuncEnum.PROCESS_ID.value])
        self._key = int(self._constant_data[PythonTracerFuncEnum.HASH_KEY.value])
        self._trace_tag = int(self._constant_data[PythonTracerFuncEnum.TRACE_TAG.value])

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
    def key(self) -> int:
        return self._key
