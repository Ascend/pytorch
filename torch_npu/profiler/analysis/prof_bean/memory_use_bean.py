import struct
from enum import Enum

from ..profiler_config import ProfilerConfig
from ..prof_common_func.constant import Constant


class MemoryEnum(Enum):
    PTR = 0
    TIME_NS = 1
    ALLOC_SIZE = 2
    TOTAL_ALLOCATED = 3
    TOTAL_RESERVED = 4
    DEVICE_TYPE = 5
    DEVICE_INDEX = 6
    THREAD_ID = 7
    PROCESS_ID = 8


class MemoryUseBean:
    CONSTANT_STRUCT = "<5qbB2Q"
    NPU_ID = 20
    CPU_ID = 0

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))

    @property
    def ptr(self) -> int:
        return int(self._constant_data[MemoryEnum.PTR.value])

    @property
    def time_us(self) -> float:
        time_us = int(self._constant_data[MemoryEnum.TIME_NS.value]) / Constant.NS_TO_US
        return ProfilerConfig().get_local_time(time_us)

    @property
    def alloc_size(self) -> int:
        return int(self._constant_data[MemoryEnum.ALLOC_SIZE.value]) / Constant.B_TO_KB

    @property
    def total_allocated(self) -> int:
        return int(self._constant_data[MemoryEnum.TOTAL_ALLOCATED.value]) / Constant.B_TO_MB

    @property
    def total_reserved(self) -> int:
        return int(self._constant_data[MemoryEnum.TOTAL_RESERVED.value]) / Constant.B_TO_MB

    @property
    def device_type(self) -> int:
        return int(self._constant_data[MemoryEnum.DEVICE_TYPE.value])

    @property
    def device_index(self) -> int:
        return int(self._constant_data[MemoryEnum.DEVICE_INDEX.value])

    @property
    def tid(self) -> int:
        return int(self._constant_data[MemoryEnum.THREAD_ID.value])

    @property
    def pid(self) -> int:
        return int(self._constant_data[MemoryEnum.PROCESS_ID.value])

    def is_npu(self) -> bool:
        return self.device_type == self.NPU_ID

    @property
    def device_tag(self) -> str:
        if self.is_npu():
            return f"NPU:{self.device_index}"
        else:
            return f"CPU"

    @property
    def row(self) -> list:
        return [Constant.PTA, self.time_us, self.total_allocated, self.total_reserved, self.device_tag]
