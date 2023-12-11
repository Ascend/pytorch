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

from warnings import warn
from math import ceil

from .base_parser import BaseParser
from ..prof_common_func.file_tag import FileTag
from ..prof_common_func.path_manager import ProfilerPathManager
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_bean.memory_use_bean import MemoryUseBean
from ..prof_common_func.constant import Constant, print_error_msg
from ..prof_common_func.constant import convert_ns2us_float, convert_ns2us_str


class MemoryPrepareParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.pta_record_list = []
        self.memory_data = []
        self._torch_op_node = []

    @staticmethod
    def _check_whether_invalid_match(allocate_record: MemoryUseBean, release_record: MemoryUseBean):
        if allocate_record.alloc_size < 0 or release_record.alloc_size > 0:
            return True
        if abs(allocate_record.alloc_size) != abs(release_record.alloc_size):
            warn(f"Memory records matching fails, alloc_sizes are "
                 f"{allocate_record.alloc_size} and {release_record.alloc_size}")
            return True
        return False

    @staticmethod
    def _find_torch_ops_by_binary_search(ts: int, torch_ops: list):
        right = len(torch_ops) - 1
        left = 0
        while right > left:
            mid = left + ceil((right - left) / 2)
            if ts >= torch_ops[mid].start_time:
                left = mid
            else:
                right = mid - 1
        return left

    def run(self, deps_data: dict):
        try:
            self._torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate operator_memory.csv or memory_record.csv.")
            return Constant.FAIL, {}
        return Constant.SUCCESS, {"pta_record_list": self.pta_record_list, "memory_data": self.memory_data}

    def generate_view(self) -> None:
        self._init_torch_op()
        self._add_pta_memory_data()

    def _find_matched_torch_op_name(self, mem_start_ts: int, torch_ops: list) -> str:
        matched_torch_op_idx = self._find_torch_ops_by_binary_search(mem_start_ts, torch_ops)
        matched_torch_op = torch_ops[matched_torch_op_idx]
        while matched_torch_op.end_time < mem_start_ts:
            matched_torch_op = matched_torch_op.parent_node
            if not matched_torch_op or not matched_torch_op.event:
                warn(f"Can't find matched torch ops for a memory record!")
                return ""
        return matched_torch_op.name

    def _combine_memory_record(self: any, allocate_record: MemoryUseBean,
                               release_record: MemoryUseBean, torch_ops: list) -> list:
        if not allocate_record:
            return ["", release_record.alloc_size, None, convert_ns2us_str(release_record.time_ns, "\t"), None, None, None,
                    release_record.total_allocated, release_record.total_reserved, release_record.device_tag]
        torch_name = self._find_matched_torch_op_name(allocate_record.time_ns, torch_ops)
        if release_record:
            return [torch_name, allocate_record.alloc_size, convert_ns2us_str(allocate_record.time_ns, "\t"),
                    convert_ns2us_str(release_record.time_ns, "\t"),
                    convert_ns2us_float(release_record.time_ns - allocate_record.time_ns),
                    allocate_record.total_allocated, allocate_record.total_reserved, release_record.total_allocated,
                    release_record.total_reserved, allocate_record.device_tag]
        else:
            return [torch_name, allocate_record.alloc_size, convert_ns2us_str(allocate_record.time_ns, "\t"), None, None,
                    allocate_record.total_allocated, allocate_record.total_reserved, None, None,
                    allocate_record.device_tag]

    def _add_pta_memory_data(self):
        pta_memory_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.MEMORY)
        pta_memory_dict = {}
        torch_op_dict = {}
        pta_memory_record = []
        pta_memory_data = sorted(pta_memory_data, key=lambda x: x.time_ns)
        for memory_re in pta_memory_data:
            if memory_re.is_npu():
                pta_memory_dict.setdefault(memory_re.pid, []).append(memory_re)
                self.pta_record_list.append(memory_re)
        for torch_op in self._torch_op_node:
            torch_op_dict.setdefault(torch_op.pid, []).append(torch_op)
        for pid_key in pta_memory_dict:
            memory_records = pta_memory_dict.get(pid_key, [])
            torch_ops = torch_op_dict.get(pid_key, [])
            if not torch_ops:
                warn(f"Lack of torch ops to connect memory record, whose process id is {pid_key}")
                continue
            torch_ops = sorted(torch_ops, key=lambda x: x.start_time)
            memory_dict = {}
            for memory_record in memory_records:
                if memory_record.ptr not in memory_dict or \
                        self._check_whether_invalid_match(memory_dict.get(memory_record.ptr), memory_record):
                    memory_dict[memory_record.ptr] = memory_record
                else:
                    pta_memory_record.append(
                        self._combine_memory_record(memory_dict.get(memory_record.ptr), memory_record, torch_ops))
                    del memory_dict[memory_record.ptr]
            for memory_record in memory_dict.values():
                if memory_record.alloc_size > 0:
                    pta_memory_record.append(self._combine_memory_record(memory_record, None, torch_ops))
                else:
                    pta_memory_record.append(self._combine_memory_record(None, memory_record, torch_ops))
        self.memory_data.extend(pta_memory_record)

    def _init_torch_op(self):
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            self._torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(only_fwk=True)
        if self._torch_op_node:
            self._torch_op_node = self._torch_op_node[1:]
