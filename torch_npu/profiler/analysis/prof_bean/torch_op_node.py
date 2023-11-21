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

from math import ceil

from ..prof_common_func.constant import Constant


class TorchOpNode:
    def __init__(self, event=None, parent_node=None, all_node_num=None):
        self._event = event
        self._parent_node = parent_node
        self._all_node_num = all_node_num
        self._child_list = []
        self._corr_id_total = []
        self._corr_id_self = []

    @property
    def event(self):
        return self._event

    @property
    def pid(self):
        return self._event.pid

    @property
    def name(self):
        return self._event.name

    @property
    def all_node_num(self):
        return self._all_node_num

    @property
    def input_shape(self):
        return self._event.args.get(Constant.INPUT_SHAPES, "")

    @property
    def call_stack(self):
        return self._event.args.get(Constant.CALL_STACK, "")

    @property
    def kernel_list(self):
        return self._kernel_list

    @property
    def start_time(self) -> int:
        return self._event.ts

    @property
    def end_time(self) -> int:
        return self._event.ts + self._event.dur

    @property
    def host_self_dur(self):
        # Time unit is ns
        return self._event.dur - sum([node.event.dur for node in self._child_list])

    @property
    def host_total_dur(self):
        # Time unit is ns
        return self._event.dur

    @property
    def child_node_list(self) -> list:
        return self._child_list

    @property
    def parent_node(self) -> any:
        return self._parent_node

    @property
    def corr_id_total(self) -> any:
        return self._corr_id_total

    @property
    def corr_id_self(self) -> any:
        return self._corr_id_self

    def is_profiler_step(self) -> bool:
        return self._event.name.find("ProfilerStep#") != -1

    def add_child_node(self, child_node):
        self._child_list.append(child_node)

    def match_child_node(self, ts_time: int) -> any:
        if not self._child_list:
            return None
        right = len(self._child_list) - 1
        left = 0
        while right > left:
            mid = left + ceil((right - left) / 2)
            if ts_time >= self._child_list[mid].start_time:
                left = mid
            else:
                right = mid - 1
        return self._child_list[left] if self._child_list[left].end_time > ts_time else None

    def update_corr_id_total(self, corr_id: int):
        self._corr_id_total.append(corr_id)

    def update_corr_id_self(self, corr_id: int):
        self._corr_id_self.append(corr_id)
