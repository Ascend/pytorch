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

__all__ = []


class OpStatisticBean(CommonBean):
    HEADERS = ["OP Type", "Core Type", "Count", "Total Time(us)", "Min Time(us)", "Avg Time(us)",
               "Max Time(us)", "Ratio(%)"]

    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def row(self) -> list:
        return list(self._data.values())

    @property
    def headers(self) -> list:
        return list(self._data.keys())
