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

from ..prof_common_func.constant import Constant


class GeMemoryRecordBean:

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        return [Constant.GE, self.time_us, self.total_allocated, self.total_reserved, self.device_tag]

    @property
    def component(self) -> str:
        return self._data.get("Component")

    @property
    def time_us(self) -> float:
        return float(self._data.get("Timestamp(us)"))

    @property
    def total_allocated(self) -> float:
        return float(self._data.get("Total Allocated(KB)", 0)) / Constant.KB_TO_MB

    @property
    def total_reserved(self) -> float:
        return float(self._data.get("Total Reserved(KB)", 0)) / Constant.KB_TO_MB

    @property
    def device_tag(self) -> float:
        return self._data.get("Device", "")
