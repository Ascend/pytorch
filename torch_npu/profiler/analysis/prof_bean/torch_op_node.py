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


class TorchOpNode:
    def __init__(self, event=None, parent_node=None, all_node_num=None):
        self._event = event
        self._parent_node = parent_node
        self._all_node_num = all_node_num
        self._child_list = []
        self._device_self_dur = 0
        self._device_self_dur_with_ai_core = 0
        self._device_total_dur = 0
        self._device_total_dur_with_ai_core = 0
        self._first_kernel_ts = 0
        self._end_kernel_ts = 0
        self._acl_ts = None

    @property
    def event(self):
        return self._event

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
    def acl_ts(self):
        return self._acl_ts

    @property
    def start_time(self) -> float:
        return self._event.ts

    @property
    def end_time(self) -> float:
        return self._event.ts + self._event.dur

    @property
    def host_self_dur(self):
        return self._event.dur - sum([node.event.dur for node in self._child_list])

    @property
    def host_total_dur(self):
        return self._event.dur

    @property
    def device_self_dur(self):
        return self._device_self_dur

    @property
    def device_self_dur_with_ai_core(self):
        return self._device_self_dur_with_ai_core

    @property
    def device_total_dur(self):
        return self._device_total_dur

    @property
    def device_total_dur_with_ai_core(self):
        return self._device_total_dur_with_ai_core

    @property
    def child_node_list(self) -> list:
        return self._child_list

    @property
    def parent_node(self) -> any:
        return self._parent_node

    @property
    def first_kernel_ts(self) -> float:
        return self._first_kernel_ts

    @property
    def end_kernel_ts(self) -> float:
        return self._end_kernel_ts

    def is_profiler_step(self) -> bool:
        return self._event.name.find("ProfilerStep#") != -1

    def add_child_node(self, child_node):
        self._child_list.append(child_node)

    def match_child_node(self, ts_time: float) -> any:
        matched_node = None
        for child_node in self._child_list:
            if child_node.start_time <= ts_time <= child_node.end_time:
                matched_node = child_node
                break
        return matched_node

    def add_device_self_dur(self, dur: float):
        self._device_self_dur += dur

    def add_device_self_dur_with_ai_core(self, dur: float):
        self._device_self_dur_with_ai_core += dur

    def add_device_total_dur(self, dur: float):
        self._device_total_dur += dur

    def add_device_total_dur_with_ai_core(self, dur: float):
        self._device_total_dur_with_ai_core += dur

    def update_first_kernel_ts(self, ts: float):
        if self._first_kernel_ts == 0:
            self._first_kernel_ts = ts
        else:
            self._first_kernel_ts = min(self._first_kernel_ts, ts)

    def update_end_kernel_ts(self, ts: float):
        self._end_kernel_ts = max(self._end_kernel_ts, ts)

    def add_acl_ts(self, acl_ts: float):
        self._acl_ts = acl_ts
