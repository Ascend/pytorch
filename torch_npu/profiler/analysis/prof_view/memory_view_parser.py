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

from ..prof_view.base_view_parser import BaseViewParser
from ..prof_common_func.file_tag import FileTag
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_common_func.file_manager import FileManager
from ..prof_bean.memory_use_bean import MemoryUseBean
from ..prof_common_func.constant import Constant
from ..prof_bean.npu_mem_bean import NpuMemoryBean
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum


class MemoryViewParser(BaseViewParser):
    HEADERS_FORM = ["Tag", "Operator", "Size(KB)", "Allocation Time(us)", "Release Time(us)", "Duration(us)",
                    "Device Type"]
    HEADERS_LINE_CHART = ["Tag", "Timestamp(us)", "Total Reserved(KB)", "Total Allocated(KB)", "Device Type"]
    MEMORY_VIEW_FORM = "memory_view_form.csv"
    MEMORY_VIEW_LINE_CHART = "memory_view_line_chart.csv"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)
        self.size_record_list = []

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
    def _find_torch_ops_by_binary_search(ts: float, torch_ops: list):
        right = len(torch_ops) - 1
        left = 0
        while right > left + 1:
            mid = left + (right - left) // 2
            if ts >= torch_ops[mid].ts:
                left = mid
            else:
                right = mid - 1
        return left

    def generate_view(self: any, output_path: str = None) -> None:
        self.size_record_list.extend(self._get_npu_memory_from_cann())
        memory_data = []
        self._add_pta_memory_data(memory_data)
        FileManager.create_csv_file(self._profiler_path, memory_data, self.MEMORY_VIEW_FORM, self.HEADERS_FORM)
        FileManager.create_csv_file(self._profiler_path, self.size_record_list,
                                    self.MEMORY_VIEW_LINE_CHART, self.HEADERS_LINE_CHART)

    def _get_npu_memory_from_cann(self: any) -> list:
        npu_memory_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.NPU_MEMORY)
        npu_memory = []
        if npu_memory_file_set:
            # all iterations are same for npu_mem.csv
            sub_file = next(iter(npu_memory_file_set))
            all_data = FileManager.read_csv_file(sub_file, NpuMemoryBean)
            for data in all_data:
                if data.row[0] != Constant.APP:
                    continue
                npu_memory.append(data.row)
        return npu_memory

    def _combine_memory_record(self: any, allocate_record: MemoryUseBean,
                               release_record: MemoryUseBean, torch_ops: list) -> list:
        start_allocate_ts = allocate_record.time_us
        if not len(torch_ops):
            warn(f"Lack of torch ops to connect memory record, "
                 f"whose process id is {allocate_record.pid} and thread id is {allocate_record.tid}.")
            return []
        matched_torch_op_idx = self._find_torch_ops_by_binary_search(start_allocate_ts, torch_ops)
        matched_torch_op = torch_ops[matched_torch_op_idx]
        device_tag = 'None'
        if allocate_record.is_npu():
            device_tag = f"NPU:{allocate_record.device_index}"
        if release_record:
            return [Constant.PTA, matched_torch_op.name, allocate_record.alloc_size / Constant.B_TO_KB,
                    allocate_record.time_us, release_record.time_us,
                    release_record.time_us - allocate_record.time_us, device_tag]
        else:
            return [Constant.PTA, matched_torch_op.name, allocate_record.alloc_size / Constant.B_TO_KB,
                    allocate_record.time_us, None, None, device_tag]

    def _add_pta_memory_data(self, memory_data_list: list):
        memory_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.MEMORY)
        torch_op_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.TORCH_OP)
        pta_memory_dict = {}
        torch_op_dict = {}
        pta_memory_record = []
        for memory_re in memory_data:
            device_tag = f"NPU:{memory_re.device_index}"
            if memory_re.is_npu():
                pta_memory_dict.setdefault(memory_re.pid, []).append(memory_re)
                self.size_record_list.append([Constant.PTA, memory_re.time_us,
                                             memory_re.total_reserved / Constant.B_TO_KB,
                                             memory_re.total_allocated / Constant.B_TO_KB,
                                             device_tag])
        for torch_op in torch_op_data:
            torch_op_dict.setdefault(torch_op.pid, []).append(torch_op)
        for pid_key in pta_memory_dict:
            memory_records = pta_memory_dict.get(pid_key, [])
            torch_ops = torch_op_dict.get(pid_key, [])
            torch_ops = sorted(torch_ops, key=lambda x: x.ts)
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
        memory_data_list.extend(pta_memory_record)
