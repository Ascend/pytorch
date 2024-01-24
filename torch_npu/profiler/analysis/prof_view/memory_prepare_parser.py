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

from collections import defaultdict
from warnings import warn
from math import ceil

from .base_parser import BaseParser
from ..prof_common_func.file_tag import FileTag
from ..prof_common_func.path_manager import ProfilerPathManager
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_bean.memory_use_bean import MemoryUseBean
from ..prof_common_func.constant import Constant, print_error_msg, print_warn_msg
from ..prof_common_func.constant import convert_ns2us_float, convert_ns2us_str


class MemoryPrepareParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.pta_record_list = []
        self.memory_data = []
        self._torch_op_node = []
        self._incomplete_num = 0

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
        if self._incomplete_num > 0:
            print_warn_msg(f"{self._incomplete_num} memory record(s) are incomplete.")
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

    def _add_pta_memory_data(self):
        pta_memory_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.MEMORY)
        npu_memory_dict = {}
        torch_op_dict = {}
        npu_memory_record = []
        pta_memory_data = sorted(pta_memory_data, key=lambda x: x.time_ns)
        for record in pta_memory_data:
            if record.is_npu():
                npu_memory_dict.setdefault(record.pid, []).append(record)
                self.pta_record_list.append(record)
        for torch_op in self._torch_op_node:
            torch_op_dict.setdefault(torch_op.pid, []).append(torch_op)
        for pid_key, memory_records in npu_memory_dict.items():
            torch_ops = torch_op_dict.get(pid_key, [])
            if not torch_ops:
                warn(f"Lack of torch ops to connect memory record, whose process id is {pid_key}")
                continue
            torch_ops = sorted(torch_ops, key=lambda x: x.start_time)
            memory_dict = defaultdict(list)
            for record in memory_records:
                memory_dict[record.ptr].append(record)
            pid_mem_buf = list()
            for ptr_records in memory_dict.values():
                ptr_records.sort(key=lambda x: x.time_ns)
                valid_record_list = self._get_valid_record_entry(ptr_records)
                pid_mem_buf.extend(valid_record_list)
            pid_mem_buf.sort(key=lambda x: x[0].time_ns)
            complete_records = self._complete_record_entry(pid_mem_buf, torch_ops)
            self.memory_data.extend(complete_records)

    @staticmethod
    def _get_valid_record_entry(records: list) -> list:
        ret_list = list()
        l_idx = r_idx = 0
        data_buf = list()
        type_buf = list()
        while l_idx < len(records) and r_idx < len(records):
            if records[l_idx].data_type != 0:
                l_idx += 1
                r_idx = l_idx
            else:
                if len(data_buf) < Constant.PTA_RECORD_TYPE_NUM and records[r_idx].data_type not in type_buf:
                    type_buf.append(records[r_idx].data_type)
                    data_buf.append(records[r_idx])
                    r_idx += 1
                else:
                    ret_list.append(data_buf[:])
                    data_buf.clear()
                    type_buf.clear()
                    l_idx = r_idx
        if data_buf:
            ret_list.append(data_buf[:])
        return ret_list

    def _complete_record_entry(self, ptr_records: list, torch_ops: list) -> list:
        ret_list = list()
        for records in ptr_records:
            if not records:
                continue
            combine_data = list()
            records_len = len(records)
            op_name = self._find_matched_torch_op_name(records[0].time_ns, torch_ops)
            if records_len == 1:
                self._incomplete_num += 2
                combine_data = [op_name, records[0].alloc_size, convert_ns2us_str(records[0].time_ns, "\t"), None, None, None, None,
                                records[0].total_allocated, records[0].total_reserved, records[0].total_active,
                                None, None, None,
                                records[0].stream_ptr, records[0].device_tag]
            elif records_len == 2:
                self._incomplete_num += 1
                active_release_time = convert_ns2us_str(records[1].time_ns, "\t") if records[1].data_type == 2 else None
                release_time = convert_ns2us_str(records[1].time_ns, "\t") if records[1].data_type == 1 else None
                duration_time = convert_ns2us_str(records[1].time_ns - records[0].time_ns, "\t") if records[1].data_type == 1 else None
                active_duration_time = convert_ns2us_str(records[1].time_ns - records[0].time_ns, "\t") if records[1].data_type == 2 else None
                combine_data = [op_name, records[0].alloc_size, convert_ns2us_str(records[0].time_ns, "\t"), release_time, active_release_time, duration_time,
                                active_duration_time, records[0].total_allocated, records[0].total_reserved, records[0].total_active,
                                records[1].total_allocated, records[1].total_reserved, records[1].total_active,
                                records[0].stream_ptr, records[0].device_tag]
            elif records_len == 3:
                free_idx = 1 if records[1].data_type == 1 else 2
                active_idx = 1 if free_idx == 2 else 2
                active_release_time = convert_ns2us_str(records[active_idx].time_ns, "\t")
                release_time = convert_ns2us_str(records[free_idx].time_ns, "\t")
                duration_time = convert_ns2us_str(records[free_idx].time_ns - records[0].time_ns, "\t")
                active_duration_time = convert_ns2us_str(records[active_idx].time_ns - records[0].time_ns, "\t")
                combine_data = [op_name, records[0].alloc_size, convert_ns2us_str(records[0].time_ns, "\t"), release_time, active_release_time, duration_time,
                                active_duration_time, records[0].total_allocated, records[0].total_reserved, records[0].total_active,
                                records[free_idx].total_allocated, records[free_idx].total_reserved, records[free_idx].total_active,
                                records[0].stream_ptr, records[0].device_tag]
            ret_list.append(combine_data[:])
        return ret_list

    def _init_torch_op(self):
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            self._torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(only_fwk=True)
        if self._torch_op_node:
            self._torch_op_node = self._torch_op_node[1:]
