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

from typing import Union, Optional
from decimal import Decimal

from ..prof_common_func.constant import Constant
from ..prof_common_func.constant import convert_us2ns
from ..prof_common_func.constant import convert_ns2us_str


class GeOpMemoryBean:
    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        return [self.name, self.size, self.allocation_time, self.release_time, None, 
                self.dur, None, self.allocation_total_allocated, self.allocation_total_reserved, None,
                self.release_total_allocated, self.release_total_reserved, None, None, self.device]

    @property
    def name(self):
        return "cann::" + self._data.get("Name")

    @property
    def size(self):
        return self._data.get("Size(KB)")

    @property
    def allocation_time(self) -> Union[float, str]:
        return self._data.get("Allocation Time(us)")

    @property
    def dur(self) -> Union[float, str]:
        return self._data.get("Duration(us)")

    @property
    def release_time(self) -> Optional[str]:
        if self.allocation_time and self.dur:
            alloc_ns = convert_us2ns(self.allocation_time)
            dur_ns = convert_us2ns(self.dur)
            rls_us = convert_ns2us_str(alloc_ns + dur_ns, "\t")
            return rls_us
        return None

    @property
    def allocation_total_allocated(self):
        size_kb = self._data.get("Allocation Total Allocated(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def allocation_total_reserved(self):
        size_kb = self._data.get("Allocation Total Reserved(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def release_total_allocated(self):
        size_kb = self._data.get("Release Total Allocated(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def release_total_reserved(self):
        size_kb = self._data.get("Release Total Reserved(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def device(self):
        return self._data.get("Device")
