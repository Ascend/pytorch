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

from .common_bean import CommonBean
from ..prof_common_func.csv_headers import CsvHeaders


class OpSummaryBean(CommonBean):
    headers = []

    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def row(self) -> list:
        row = []
        read_headers = OpSummaryBean.headers if OpSummaryBean.headers else self._data.keys()
        for field_name in read_headers:
            row.append(self._data.get(field_name, ""))
        return row

    @property
    def ts(self) -> str:
        # Time us str
        return self._data.get(CsvHeaders.TASK_START_TIME, "0")

    @property
    def all_headers(self) -> list:
        return list(self._data.keys())
