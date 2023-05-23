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

class NodeInfoBean:
    def __init__(self, acl_start_time: float, kernel_list: list):
        self._acl_start_time = acl_start_time
        self._device_dur = 0
        self._device_dur_with_ai_core = 0
        self._kernel_min_ts = 0
        self._kernel_max_ts = 0
        self._kernel_list = kernel_list
        self._init()

    def _init(self):
        self._device_dur = sum([kernel.dur for kernel in self._kernel_list])
        self._device_dur_with_ai_core = sum([kernel.dur for kernel in self._kernel_list if kernel.is_ai_core])
        self._kernel_min_ts = min([kernel.ts for kernel in self._kernel_list])
        self._kernel_max_ts = max([kernel.ts for kernel in self._kernel_list])

    @property
    def acl_start_time(self) -> float:
        return self._acl_start_time

    @property
    def device_dur(self) -> float:
        return self._device_dur

    @property
    def device_dur_with_ai_core(self) -> float:
        return self._device_dur_with_ai_core

    @property
    def kernel_min_ts(self) -> float:
        return self._kernel_min_ts

    @property
    def kernel_max_ts(self) -> float:
        return self._kernel_max_ts
