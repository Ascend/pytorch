 # Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

import itertools
from typing import Any
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Dict, List, Tuple, Optional
import math
from enum import Enum
import torch_npu

try:
    from contextlib import ContextDecorator
except ImportError:
    import functools

    class ContextDecorator(object):  # type: ignore[no-redef]

        def __enter__(self):
            raise NotImplementedError

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError

        def __call__(self, func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapped
__all__ = ["profile", "record_function", ]

if not torch_npu._C._profiler_init():
    raise RuntimeError("proflier initialization failed")

class DeviceType(Enum):
    CPU = 0,
    CUDA = 1, # CUDA.
    MKLDNN = 2, # Reserved for explicit MKLDNN
    OPENGL = 3, # OpenGL
    OPENCL = 4, # OpenCL
    IDEEP = 5, # IDEEP.
    HIP = 6, # AMD HIP
    FPGA = 7, # FPGA
    MSNPU = 8, # MSNPU
    XLA = 9, # XLA / TPU
    Vulkan = 10, # Vulkan
    Metal = 11, # Metal
    XPU = 12, # XPU
    NPU = 13, # NPU
class EventList(list):
    """A list of Events (for pretty printing)"""
    def __init__(self, *args, **kwargs):
        use_cuda = False
        use_npu = kwargs.pop('use_npu', True) and torch_npu.npu.is_available()
        profile_memory = kwargs.pop('profile_memory', False)
        with_flops = kwargs.pop('with_flops', False)
        super(EventList, self).__init__(*args, **kwargs)
        self._use_cuda = use_cuda
        self._use_npu = use_npu
        self._profile_memory = profile_memory
        self._tree_built = False
        self._with_flops = with_flops
        assert not (self._use_cuda and self._use_npu), "use_cuda and use_npu can't be True simultaneously."

    def _build_tree(self):
        self._populate_cpu_children()
        self._remove_dup_nodes()
        self._set_backward_stacktraces()
        self._tree_built = True

    def __str__(self):
        return self.table()

    def _remove_dup_nodes(self):
        while True:
            to_delete = []
            for idx in range(len(self)):
                if (self[idx].cpu_parent is not None and
                        self[idx].cpu_parent.name == self[idx].name and
                        len(self[idx].cpu_parent.cpu_children) == 1):
                    self[idx].cpu_parent.cpu_children = self[idx].cpu_children
                    self[idx].cpu_parent.kernels = self[idx].kernels  # lift kernels up
                    for ch in self[idx].cpu_children:
                        ch.cpu_parent = self[idx].cpu_parent
                    to_delete.append(idx)
            if len(to_delete) == 0:
                break
            new_evts = [ev for ind, ev in enumerate(self) if ind not in to_delete]
            self.clear()
            self.extend(new_evts)

    def _populate_cpu_children(self):
        """Populates child events into each underlying FunctionEvent object.
        One event is a child of another if [s1, e1) is inside [s2, e2). Where
        s1 and e1 would be start and end of the child event's interval. And
        s2 and e2 start and end of the parent event's interval

        Example: In event list [[0, 10], [1, 3], [3, 4]] would have make [0, 10]
        be a parent of two other intervals.

        If for any reason two intervals intersect only partially, this function
        will not record a parent child relationship between then.
        """

        # Some events can be async (i.e. start and end on different threads),
        # since it's generally undefined how to attribute children ranges to
        # async ranges, we do not use them when calculating nested ranges and stats
        sync_events = [evt for evt in self if not evt.is_async and evt.device_type == DeviceType.CPU]
        events = sorted(
            sync_events,
            key=attrgetter("thread"),
        )
        # Group by both thread and node_id, so that events that happen to have
        # the same thread_id but are from different nodes aren't incorrectly
        # grouped together.
        threads = itertools.groupby(
            events, key=lambda event: (event.thread, event.node_id)
        )

        # For each thread we keep a stack of current nested parents.
        # We maintain the invariant that each interval is a subset of all other
        # intervals lower in the stack.
        #
        # First we sort the intervals by their start time. Then we iterate over them.
        # Every time we see a new interval we remove several parents from
        # the top until we restore the invariant. Then parent child relationship
        # if recorded if the stack is not empty.
        # Finally we add new interval to the list
        #
        # Algorithm has O(N * log(N)) complexity where N is number of
        # intervals
        for thread_id, thread_events in threads:
            thread_events_ = sorted(
                thread_events,
                key=lambda event: [event.time_range.start, -event.time_range.end],
            )
            current_events: List[FunctionEvent] = []
            cur_end = 0
            for event in thread_events_:
                while len(current_events) > 0:
                    parent = current_events[-1]
                    if event.time_range.start >= parent.time_range.end or \
                            event.time_range.end > parent.time_range.end:
                        # this can't be a parent
                        current_events.pop()
                    else:
                        parent.append_cpu_child(event)
                        assert (
                            event.cpu_parent is None
                        ), "There is already a CPU parent event for {}".format(
                            event.key
                        )
                        event.set_cpu_parent(parent)
                        break

                current_events.append(event)

    def _set_backward_stacktraces(self):
        def bw_parent(evt):
            if evt is None:
                return None
            elif evt.scope == 1:  # BACKWARD_FUNCTION
                return evt
            else:
                return bw_parent(evt.cpu_parent)

        fwd_stacks = {}
        for evt in self:
            if bw_parent(evt) is None and evt.stack is not None:
                t = (evt.sequence_nr, evt.thread)
                if t not in fwd_stacks:
                    fwd_stacks[t] = evt.stack

        for evt in self:
            p = bw_parent(evt)
            if p is not None:
                assert p.fwd_thread is not None
                t = (p.sequence_nr, p.fwd_thread)
                if t in fwd_stacks:
                    evt.stack = fwd_stacks[t]
                else:
                    evt.stack = []

    @property
    def self_cpu_time_total(self):
        return sum([event.self_cpu_time_total for event in self])

    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, header=None, top_level_events_only=False):
        """Prints an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda ops events are omitted for profiler result readability.

        Returns:
            A string containing the table.
        """
        return build_table(
            self,
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            header=header,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
            top_level_events_only=top_level_events_only,
            use_cuda=self._use_cuda,
            use_npu=self._use_npu)

    def export_chrome_trace(self, path):
        """Exports an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
        import os
        with open(path, 'w') as f:
            chrome_events = []
            next_id = 0
            # Use file IO over using json.dump since JSON dumping is very slow and
            # this technique is proven to give a 4x speedup.
            f.write("[")
            for evt in self:
                if evt.trace_name is None:
                    continue
                f.write(
                    '{"name": "%s", '
                    '"ph": "X", '
                    '"ts": %s, '
                    '"dur": %s, '
                    '"tid": %s, '
                    '"pid": "CPU functions", '
                    '"args": {}}, '
                    % (
                        evt.trace_name,
                        evt.time_range.start,
                        evt.time_range.elapsed_us(),
                        evt.thread
                        if not evt.is_remote
                        else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "',
                    )
                )
                if self._use_cuda:
                    for k in evt.kernels:
                        # 's' and 'f' draw Flow arrows from
                        # the CPU launch to the GPU kernel
                        f.write('{"name": "%s", '
                                '"ph": "s", '
                                '"ts": %s, '
                                '"tid": %s, '
                                '"pid": "CPU functions", '
                                '"id": %s, '
                                '"cat": "cpu_to_cuda", '
                                '"args": {}}, ' % (evt.trace_name, evt.time_range.start,
                                                evt.thread, next_id))
                        f.write('{"name": "%s", '
                                '"ph": "f", '
                                '"ts": %s, '
                                '"tid": %s, '
                                '"pid": "CUDA functions", '
                                '"id": %s, '
                                '"cat": "cpu_to_cuda", '
                                '"args": {}}, ' % (k.name, k.interval.start, k.device, next_id))
                        f.write('{"name": "%s", '
                                '"ph": "X", '
                                '"ts": %s, '
                                '"dur": %s, '
                                '"tid": %s, '
                                '"pid": "CUDA functions", '
                                '"args": {}}, ' % (k.name, k.interval.start,
                                                k.interval.elapsed_us(), k.device))
                        next_id += 1
                elif self._use_npu:
                    for k in evt.kernels:
                        # 's' and 'f' draw Flow arrows from
                        # the CPU launch to the GPU kernel
                        f.write('{"name": "%s", '
                                '"ph": "s", '
                                '"ts": %s, '
                                '"tid": %s, '
                                '"pid": "CPU functions", '
                                '"id": %s, '
                                '"cat": "cpu_to_npu", '
                                '"args": {}}, ' % (evt.trace_name, evt.time_range.start,
                                                evt.thread, next_id))
                        f.write('{"name": "%s", '
                                '"ph": "f", '
                                '"ts": %s, '
                                '"tid": %s, '
                                '"pid": "NPU functions", '
                                '"id": %s, '
                                '"cat": "cpu_to_npu", '
                                '"args": {}}, ' % (k.name, k.interval.start, k.device, next_id))
                        f.write('{"name": "%s", '
                                '"ph": "X", '
                                '"ts": %s, '
                                '"dur": %s, '
                                '"tid": %s, '
                                '"pid": "NPU functions", '
                                '"args": {}}, ' % (k.name, k.interval.start,
                                                k.interval.elapsed_us(), k.device))
                        next_id += 1 

            # remove trailing whitespace and comma
            f.seek(f.tell() - 2, os.SEEK_SET)
            f.truncate()
            f.write("]")

    def supported_export_stacks_metrics(self):
        if self._use_npu:
            return ["self_cpu_time_total", "self_npu_time_total"]
        return ["self_cpu_time_total", "self_cuda_time_total"]

    def export_stacks(self, path: str, metric: str):
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError("metric should be one of: " + str(self.supported_export_stacks_metrics()))
        translate_table = str.maketrans(" ;\t\n", "____")
        with open(path, 'w') as f:
            for evt in self:
                if evt.stack and len(evt.stack) > 0:
                    metric_value = getattr(evt, metric)
                    if int(metric_value) > 0:
                        stack_str = ""
                        for entry in reversed(evt.stack):
                            stack_str += entry.translate(translate_table)
                            stack_str += ";"
                        stack_str = stack_str[:-1] + " " + str(int(metric_value))
                        f.write(stack_str + "\n")

    def key_averages(self, group_by_input_shapes=False, group_by_stack_n=0):
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
            (event name, input shapes) rather than just event name.
            This is useful to see which input shapes contribute to the runtime
            the most and may help with size-specific optimizations or
            choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
        assert self._tree_built
        stats: Dict[Tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(event, group_by_input_shapes, group_by_stack_n) -> Tuple[str, ...]:
            key = [str(event.key), str(event.node_id), str(event.device_type), str(event.is_legacy)]
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)
        for evt in self:
            stats[get_key(evt, group_by_input_shapes, group_by_stack_n)].add(evt, self._use_cuda, self._use_npu)

        avg_list = EventList(
            stats.values(),
            use_cuda=self._use_cuda,
            use_npu=self._use_npu,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops)
        for evt in avg_list:
            evt.stack = evt.stack[:group_by_stack_n]
            if not group_by_input_shapes:
                evt.input_shapes = ""
        return avg_list

    def total_average(self):
        """Averages all events.

        Returns:
            A FunctionEventAvg object.
        """
        total_stat = FunctionEventAvg()
        for evt in self:
            total_stat += evt
            total_stat.key = None
        total_stat.key = 'Total'
        return total_stat
