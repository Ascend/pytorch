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


class GeOpMemoryBean:
    HEADERS = ["Name", "Size(KB)", "Allocation Time(us)", "Release Time(us)", "Duration(us)", "Device Type",
               "Allocation Total Allocated(MB)", "Allocation Total Reserved(MB)",
               "Release Total Allocated(MB)", "Release Total Reserved(MB)"]

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        row = []
        for field_name in self.HEADERS:
            row.append(self._data.get(field_name, ""))
        return row
