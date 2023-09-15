from math import ceil

from .node_info_bean import NodeInfoBean
from ..prof_common_func.constant import Constant


class TorchOpNode:
    def __init__(self, event=None, parent_node=None, all_node_num=None):
        self._event = event
        self._parent_node = parent_node
        self._all_node_num = all_node_num
        self._child_list = []
        self._device_dur_list = [0, 0, 0, 0]
        self._kernel_list = []
        self._min_start = float('inf')
        self._max_end = -float('inf')

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
    def kernel_list(self):
        return self._kernel_list

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
        return self._device_dur_list[0]

    @property
    def device_self_dur_with_ai_core(self):
        return self._device_dur_list[1]

    @property
    def device_total_dur(self):
        return self._device_dur_list[2]

    @property
    def device_total_dur_with_ai_core(self):
        return self._device_dur_list[3]

    @property
    def child_node_list(self) -> list:
        return self._child_list

    @property
    def parent_node(self) -> any:
        return self._parent_node

    @property
    def device_start(self) -> any:
        return self._min_start

    @property
    def device_end(self) -> any:
        return self._max_end

    def is_profiler_step(self) -> bool:
        return self._event.name.find("ProfilerStep#") != -1

    def add_child_node(self, child_node):
        self._child_list.append(child_node)

    def match_child_node(self, ts_time: float) -> any:
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

    def update_device_self(self, node_info_bean: NodeInfoBean):
        self._device_dur_list[0] += node_info_bean.device_dur
        self._device_dur_list[1] += node_info_bean.device_dur_with_ai_core
        self._kernel_list = node_info_bean.kernel_list

    def update_device_total(self, node_info_bean: NodeInfoBean):
        self._device_dur_list[2] += node_info_bean.device_dur
        self._device_dur_list[3] += node_info_bean.device_dur_with_ai_core

    def update_device_range(self, node_info_bean: NodeInfoBean):
        self._min_start = min([self._min_start, node_info_bean.min_start])
        self._max_end = max([self._max_end, node_info_bean.max_end])
