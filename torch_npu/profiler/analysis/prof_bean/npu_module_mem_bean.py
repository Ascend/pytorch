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


class NpuModuleMemoryBean:
    SHOW_HEADERS = ["Component", "Timestamp(us)", "Total Reserved(MB)", "Device"]

    def __init__(self, data: dict):
        total_reverved = int(data.get("Total Reserved(KB)", 0))
        data["Total Reserved(MB)"] = str(total_reverved / Constant.KB_TO_MB)
        self._data = data

    @property
    def row(self) -> list:
        return [self._data.get(field_name, "") for field_name in self.SHOW_HEADERS]

    @property
    def headers(self) -> list:
        return self.SHOW_HEADERS