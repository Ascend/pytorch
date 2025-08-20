import struct
from enum import Enum

from ._common_bean import CommonBean
from .._profiler_config import ProfilerConfig
from ..prof_common_func._constant import Constant
from ..prof_common_func._constant import convert_ns2us_str

__all__ = []


class MemoryEnum(Enum):
    PTR = 0
    TIME_NS = 1
    ALLOC_SIZE = 2
    TOTAL_ALLOCATED = 3
    TOTAL_RESERVED = 4
    TOTAL_ACTIVE = 5
    STREAM_PTR = 6
    DEVICE_TYPE = 7
    DEVICE_INDEX = 8
    COMPONENT_TYPE = 9
    DATA_TYPE = 10
    ALLOCATOR_TYPE = 11
    THREAD_ID = 12
    PROCESS_ID = 13


class MemoryUseBean(CommonBean):
    CONSTANT_STRUCT = "<7q2b3B2Q"
    CONSTANT_UNPACKER = struct.Struct(CONSTANT_STRUCT)
    NPU_ID = 20
    CPU_ID = 0
    INNER_ALLOCATOR = 0

    def __init__(self, data: dict):
        super().__init__(data)
        self._constant_data = self.CONSTANT_UNPACKER.unpack(data.get(Constant.CONSTANT_BYTES))
        self._ptr = self._constant_data[MemoryEnum.PTR.value]
        self._stream_ptr = self._constant_data[MemoryEnum.STREAM_PTR.value]
        profiler_config = ProfilerConfig()
        self._time_ns = profiler_config.get_local_time(
            profiler_config.get_timestamp_from_syscnt(self._constant_data[MemoryEnum.TIME_NS.value]))
        self._alloc_size = self._constant_data[MemoryEnum.ALLOC_SIZE.value]
        self._total_allocated = self._constant_data[MemoryEnum.TOTAL_ALLOCATED.value]
        self._total_reserved = self._constant_data[MemoryEnum.TOTAL_RESERVED.value]
        self._total_active = self._constant_data[MemoryEnum.TOTAL_ACTIVE.value]
        self._device_type = self._constant_data[MemoryEnum.DEVICE_TYPE.value]
        self._device_index = -1
        self._component_type = self._constant_data[MemoryEnum.COMPONENT_TYPE.value]
        self._data_type = self._constant_data[MemoryEnum.DATA_TYPE.value]
        self._allocator_type = self._constant_data[MemoryEnum.ALLOCATOR_TYPE.value]
        self._thread_id = self._constant_data[MemoryEnum.THREAD_ID.value]
        self._process_id = self._constant_data[MemoryEnum.PROCESS_ID.value]

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def stream_ptr(self) -> int:
        return self._stream_ptr

    @property
    def time_ns(self) -> int:
        return self._time_ns

    @property
    def alloc_size(self) -> float:
        return self._alloc_size / Constant.B_TO_KB

    @property
    def alloc_size_for_db(self) -> int:
        return self._alloc_size

    @property
    def total_allocated(self) -> float:
        return self._total_allocated / Constant.B_TO_MB

    @property
    def total_allocated_for_db(self) -> int:
        return self._total_allocated

    @property
    def total_reserved(self) -> float:
        return self._total_reserved / Constant.B_TO_MB

    @property
    def total_reserved_for_db(self) -> int:
        return self._total_reserved

    @property
    def total_active(self) -> float:
        return self._total_active / Constant.B_TO_MB

    @property
    def total_active_for_db(self) -> int:
        return self._total_active

    @property
    def device_type(self) -> int:
        return self._device_type

    @property
    def device_index(self) -> int:
        return (self._device_index if self._device_index != -1 else
                int(self._constant_data[MemoryEnum.DEVICE_INDEX.value]))

    @device_index.setter
    def device_index(self, value: int) -> None:
        self._device_index = value

    @property
    def component_type(self) -> int:
        return self._component_type

    @property
    def data_type(self) -> int:
        return self._data_type
    
    @property
    def allocator_type(self) -> int:
        return self._allocator_type

    @property
    def tid(self) -> int:
        return self._thread_id

    @property
    def pid(self) -> int:
        return self._process_id

    def is_npu(self) -> bool:
        return self.device_type == self.NPU_ID

    def is_inner_allocator(self) -> bool:
        return self.allocator_type == self.INNER_ALLOCATOR

    @property
    def device_tag(self) -> str:
        if self.is_npu():
            return f"NPU:{self.device_index}"
        else:
            return f"CPU"

    @property
    def row(self) -> list:
        return [Constant.PTA, convert_ns2us_str(self.time_ns, tail="\t"), self.total_allocated,
                self.total_reserved, self.total_active, self.stream_ptr, self.device_tag]
