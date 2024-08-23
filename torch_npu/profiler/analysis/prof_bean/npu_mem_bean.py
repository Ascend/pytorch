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


class NpuMemoryBean:
    SHOW_HEADERS = ["event", "timestamp(us)", "allocated(KB)", "memory(KB)", "active", "stream_ptr", "Device_id"]

    def __init__(self, data: dict):
        self._data = data

    @property
    def row(self) -> list:
        row = []
        if self._data.get("event") != Constant.APP:
            return row
        for field_name in self.SHOW_HEADERS:
            if field_name == "memory(KB)":
                row.append(float(self._data.get(field_name, 0)) / Constant.KB_TO_MB)
            else:
                row.append(self._data.get(field_name, ""))
        return row