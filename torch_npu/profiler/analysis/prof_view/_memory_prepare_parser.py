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

from ._base_parser import BaseParser
from ..prof_common_func._file_tag import FileTag
from ..prof_common_func._path_manager import ProfilerPathManager
from ..prof_parse._fwk_file_parser import FwkFileParser
from ..prof_bean._memory_use_bean import MemoryUseBean
from ..prof_bean._op_mark_bean import OpMarkBean
from ..prof_common_func._constant import Constant, print_warn_msg
from ..prof_common_func._constant import convert_ns2us_float, convert_ns2us_str
from ..prof_common_func._log import ProfilerLogger
from .._profiler_config import ProfilerConfig

__all__ = []
TASK_QUEUE_ENABLE = 'TASK_QUEUE_ENABLE'
ATEN_OP_NAME_PREFIX = 'aten'


class MemoryPrepareParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.pta_record_list = []
        self.memory_data = dict()
        self._torch_op_node = []
        self._incomplete_num = 0
        self._is_malloc_workspace_in_dequeue_enabled = False
        self._dequeue_record_dict = defaultdict(list)  # {(pid, tid): [dequeue_records]}
        self._enqueue_record_dict = {}  # {corrid: enqueue}
        self._dequeue_pids = set()
        self._dequeue_tids = set()
        ProfilerLogger.init(self._profiler_path, "MemoryPrepareParser")
        self.logger = ProfilerLogger.get_instance()

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
        except Exception as e:
            self.logger.error("Failed to generate pytorch memory data, error: %s", str(e), exc_info=True)
            return Constant.FAIL, {}
        if self._incomplete_num > 0:
            print_warn_msg(f"{self._incomplete_num} memory record(s) are incomplete.")
        return Constant.SUCCESS, {"pta_record_list": self.pta_record_list, "memory_data": self.memory_data}

    def generate_view(self) -> None:
        ProfilerConfig().load_info(self._profiler_path)
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

    def _init_queue_info(self):
        enqueue_records = FwkFileParser(self._profiler_path).get_enqueue_data()
        for enqueue_record in enqueue_records:
            self._enqueue_record_dict[enqueue_record.corr_id] = enqueue_record
        dequeue_records = FwkFileParser(self._profiler_path).get_dequeue_data()
        for dequeue_record in dequeue_records:
            self._dequeue_pids.add(dequeue_record.pid)
            self._dequeue_tids.add(dequeue_record.tid)
            key = (dequeue_record.pid, dequeue_record.tid)
            self._dequeue_record_dict.setdefault(key, []).append(dequeue_record)

    def _add_pta_memory_data(self):
        self._init_queue_info()
        pta_memory_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.MEMORY)
        npu_memory_dict = {}
        torch_op_dict = {}
        pta_memory_data = sorted(pta_memory_data, key=lambda x: x.time_ns)
        for record in pta_memory_data:
            if record.is_npu():
                if record.is_inner_allocator():
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
            if Constant.Text in self._export_type:
                self.memory_data.setdefault(Constant.Text, self._complete_record_entry(pid_mem_buf, torch_ops))
            if Constant.Db in self._export_type:
                self.memory_data.setdefault(Constant.Db, self._complete_record_entry_for_db(pid_mem_buf, torch_ops))

    @staticmethod
    def _get_valid_record_entry(records: list) -> list:
        ret_list = list()
        l_idx = r_idx = 0
        data_buf = list()
        type_buf = list()
        while l_idx < len(records) and r_idx < len(records):
            if records[l_idx].data_type != Constant.MEMORY_MALLOC:
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

    def _find_dequeue_record_by_binary_search(self, ts: int, dequeue_records: list) -> int:
        right = len(dequeue_records) - 1
        left = 0
        while right > left:
            mid = left + ceil((right - left) / 2)
            if ts >= dequeue_records[mid].ts:
                left = mid
            else:
                right = mid - 1
        return left

    def _find_related_dequeue_record(self, record: MemoryUseBean) -> OpMarkBean:
        if not (record.pid in self._dequeue_pids and record.tid in self._dequeue_tids):
            return None
        dequeue_records = self._dequeue_record_dict[(record.pid, record.tid)]
        index = self._find_dequeue_record_by_binary_search(record.time_ns, dequeue_records)
        if not (dequeue_records[index].ts <= record.time_ns < dequeue_records[index].ts +
                dequeue_records[index].dur):
            warn("Cannot find dequeue record matched memory record")
            return None
        return dequeue_records[index]

    def _find_related_enqueue_record(self, dequeue_record: OpMarkBean) -> OpMarkBean:
        return self._enqueue_record_dict.get(dequeue_record.corr_id)

    def _get_aten_op_name_by_enqueue_record(self, enqueue_record: OpMarkBean, torch_ops: list) -> str:
        index = self._find_torch_ops_by_binary_search(enqueue_record.time_ns, torch_ops)
        while index >= 0 and (not torch_ops[index].name.startswith(ATEN_OP_NAME_PREFIX)):
            index = index - 1
        if index == -1:
            warn("Unable to find aten operator according to enqueue record.")
            return ""
        return torch_ops[index].name

    def _find_real_op_name_of_record(self, dequeue_record: OpMarkBean, torch_ops: list) -> str:
        enqueue_record = self._find_related_enqueue_record(dequeue_record)
        if enqueue_record is None:
            warn("Unable to find enqueue record according to dequeue record.")
            return ""
        return self._get_aten_op_name_by_enqueue_record(enqueue_record, torch_ops)

    def _complete_record_entry(self, ptr_records: list, torch_ops: list) -> list:
        ret_list = list()
        torch_ops = [torch_op for torch_op in torch_ops if torch_op.name != "empty_tensor" and torch_op.name != "malloc_workspace"]
        for records in ptr_records:
            combine_data = list()
            records_len = len(records)
            if not records or records_len > 3:
                continue
            dequeue_record = self._find_related_dequeue_record(records[0])
            if dequeue_record is None:
                op_name = self._find_matched_torch_op_name(records[0].time_ns, torch_ops)
            else:
                op_name = self._find_real_op_name_of_record(dequeue_record, torch_ops)
            if records_len == 1:
                if hasattr(records[0], 'component_type') and records[0].component_type == Constant.CACHING_TYPE:
                    self._incomplete_num += 2
                combine_data = [op_name, records[0].alloc_size, convert_ns2us_str(records[0].time_ns, "\t"), None, None, None, None,
                                records[0].total_allocated, records[0].total_reserved, records[0].total_active,
                                None, None, None,
                                records[0].stream_ptr, records[0].device_tag]
            elif records_len == 2:
                if hasattr(records[0], 'component_type') and records[0].component_type == Constant.CACHING_TYPE:
                    self._incomplete_num += 1
                active_release_time = convert_ns2us_str(records[1].time_ns, "\t") if records[1].data_type == Constant.MEMORY_BLOCK_FREE else None
                release_time = convert_ns2us_str(records[1].time_ns, "\t") if records[1].data_type == Constant.MEMORY_FREE else None
                duration_time = convert_ns2us_str(records[1].time_ns - records[0].time_ns, "\t") if records[1].data_type == Constant.MEMORY_FREE else None
                active_duration_time = convert_ns2us_str(records[1].time_ns - records[0].time_ns, "\t") if records[1].data_type == Constant.MEMORY_BLOCK_FREE else None
                combine_data = [op_name, records[0].alloc_size, convert_ns2us_str(records[0].time_ns, "\t"), release_time, active_release_time, duration_time,
                                active_duration_time, records[0].total_allocated, records[0].total_reserved, records[0].total_active,
                                records[1].total_allocated, records[1].total_reserved, records[1].total_active,
                                records[0].stream_ptr, records[0].device_tag]
            elif records_len == 3:
                free_idx = 1 if records[1].data_type == Constant.MEMORY_FREE else 2
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

    def _complete_record_entry_for_db(self, ptr_records: list, torch_ops: list) -> list:
        ret_list = list()
        torch_ops = [torch_op for torch_op in torch_ops if torch_op.name != "empty_tensor" and torch_op.name != "malloc_workspace"]
        for records in ptr_records:
            combine_data = list()
            records_len = len(records)
            if not records or records_len > 3:
                continue
            dequeue_record = self._find_related_dequeue_record(records[0])
            if dequeue_record is None:
                op_name = self._find_matched_torch_op_name(records[0].time_ns, torch_ops)
            else:
                op_name = self._find_real_op_name_of_record(dequeue_record, torch_ops)
            if records_len == 1:
                if hasattr(records[0], 'component_type') and records[0].component_type == Constant.CACHING_TYPE:
                    self._incomplete_num += 2
                combine_data = [op_name, records[0].alloc_size_for_db, records[0].time_ns, None, None, None, None,
                                records[0].total_allocated_for_db, records[0].total_reserved_for_db, records[0].total_active_for_db,
                                None, None, None,
                                records[0].stream_ptr, records[0].device_index]
            elif records_len == 2:
                if hasattr(records[0], 'component_type') and records[0].component_type == Constant.CACHING_TYPE:
                    self._incomplete_num += 1
                active_release_time = records[1].time_ns if records[1].data_type == Constant.MEMORY_BLOCK_FREE else None
                release_time = records[1].time_ns if records[1].data_type == Constant.MEMORY_FREE else None
                duration_time = records[1].time_ns - records[0].time_ns if records[1].data_type == Constant.MEMORY_FREE else None
                active_duration_time = records[1].time_ns - records[0].time_ns if records[1].data_type == Constant.MEMORY_BLOCK_FREE else None
                combine_data = [op_name, records[0].alloc_size_for_db, records[0].time_ns, release_time, active_release_time, duration_time,
                                active_duration_time, records[0].total_allocated_for_db, records[0].total_reserved_for_db, records[0].total_active_for_db,
                                records[1].total_allocated_for_db, records[1].total_reserved_for_db, records[1].total_active_for_db,
                                records[0].stream_ptr, records[0].device_index]
            elif records_len == 3:
                free_idx = 1 if records[1].data_type == Constant.MEMORY_FREE else 2
                active_idx = 1 if free_idx == 2 else 2
                duration_time = records[free_idx].time_ns - records[0].time_ns
                active_duration_time = records[active_idx].time_ns - records[0].time_ns
                combine_data = [op_name, records[0].alloc_size_for_db, records[0].time_ns, records[free_idx].time_ns, records[active_idx].time_ns, duration_time,
                                active_duration_time, records[0].total_allocated_for_db, records[0].total_reserved_for_db, records[0].total_active_for_db,
                                records[free_idx].total_allocated_for_db, records[free_idx].total_reserved_for_db, records[free_idx].total_active_for_db,
                                records[0].stream_ptr, records[0].device_index]
            ret_list.append(combine_data[:])
        return ret_list

    def _init_torch_op(self):
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            self._torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(only_fwk=True)
        if self._torch_op_node:
            self._torch_op_node = self._torch_op_node[1:]
