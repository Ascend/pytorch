# Copyright (c) 2023, Huawei Technologies.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from enum import Enum

from ..profiler_config import ProfilerConfig
from ..prof_common_func.constant import Constant
from ..prof_common_func.constant import convert_ns2us_str


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
    DATA_TYPE = 9
    THREAD_ID = 10
    PROCESS_ID = 11


class MemoryUseBean:
    CONSTANT_STRUCT = "<7qb2B2Q"
    NPU_ID = 9
    CPU_ID = 0

    def __init__(self, data: dict):
        self._origin_data = data
        self._constant_data = struct.unpack(self.CONSTANT_STRUCT, data.get(Constant.CONSTANT_BYTES))

    @property
    def ptr(self) -> int:
        return int(self._constant_data[MemoryEnum.PTR.value])

    @property
    def stream_ptr(self) -> int:
        return int(self._constant_data[MemoryEnum.STREAM_PTR.value])

    @property
    def time_ns(self) -> int:
        time_ns = ProfilerConfig().get_timestamp_from_syscnt(self._constant_data[MemoryEnum.TIME_NS.value])
        return ProfilerConfig().get_local_time(time_ns)

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
    def total_active(self) -> int:
        return int(self._constant_data[MemoryEnum.TOTAL_ACTIVE.value]) / Constant.B_TO_MB

    @property
    def device_type(self) -> int:
        return int(self._constant_data[MemoryEnum.DEVICE_TYPE.value])

    @property
    def device_index(self) -> int:
        return int(self._constant_data[MemoryEnum.DEVICE_INDEX.value])

    @property
    def data_type(self) -> int:
        return int(self._constant_data[MemoryEnum.DATA_TYPE.value])

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
        return [Constant.PTA, convert_ns2us_str(self.time_ns, tail="\t"), self.total_allocated,
                self.total_reserved, self.total_active, self.stream_ptr, self.device_tag]
