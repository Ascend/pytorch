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

class GeMemoryRecordBean:
    HEADERS = ["Component", "Timestamp(us)", "Total Allocated(MB)", "Total Reserved(MB)", "Device Type"]

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        row = []
        for field_name in self.HEADERS:
            row.append(self._data.get(field_name, ""))
        return row

    @property
    def time_us(self) -> float:
        return float(self._data.get("Timestamp(us)"))

    @property
    def total_allocated(self) -> float:
        return float(self._data.get("Total Allocated(MB)", 0))

    @property
    def total_reserved(self) -> float:
        return float(self._data.get("Total Reserved(MB)", 0))

    @property
    def device_tag(self) -> float:
        return self._data.get("Device Type", "")
