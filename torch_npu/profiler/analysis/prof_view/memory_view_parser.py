from warnings import warn
from math import ceil
import os

from ..prof_view.base_view_parser import BaseViewParser
from ..prof_common_func.file_tag import FileTag
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_common_func.file_manager import FileManager
from ..prof_bean.memory_use_bean import MemoryUseBean
from ..prof_common_func.constant import Constant
from ..prof_bean.npu_mem_bean import NpuMemoryBean
from ..prof_bean.ge_op_memory_bean import GeOpMemoryBean
from ..prof_bean.ge_memory_record_bean import GeMemoryRecordBean
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum


class MemoryViewParser(BaseViewParser):
    HEADERS_OPERATOR = ["Name", "Size(KB)", "Allocation Time(us)", "Release Time(us)", "Duration(us)",
                        "Allocation Total Allocated(MB)", "Allocation Total Reserved(MB)",
                        "Release Total Allocated(MB)", "Release Total Reserved(MB)", "Device Type"]
    HEADERS_RECORD = ["Component", "Timestamp(us)", "Total Allocated(MB)", "Total Reserved(MB)", "Device Type"]
    OPERATOR_MEMORY = "operator_memory.csv"
    MEMORY_RECORD = "memory_record.csv"
    MAX_FIND_LAYERS = 100

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)
        self.size_record_list = []
        self.pta_record_list = []
        self.ge_record_list = []
        self.memory_data = []
        self.component_list = []

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
        while right > left:
            mid = left + ceil((right - left) / 2)
            if ts >= torch_ops[mid].ts:
                left = mid
            else:
                right = mid - 1
        return left

    @staticmethod
    def _get_data_from_file(file_set: set, file_type_bean: any, bean_list: bool = False) -> list:
        data_list = []
        if file_set:
            # only need to read one file if there exist more than one files
            sub_file = next(iter(file_set))
            all_data = FileManager.read_csv_file(sub_file, file_type_bean)
            if bean_list:
                return all_data
            for data in all_data:
                if data.row:
                    data_list.append(data.row)
        return data_list

    @staticmethod
    def _combine_record(last_record, cur_record):
        cur_record_list = cur_record.row
        if last_record:
            pta_ge_record_list = [Constant.PTA_GE, cur_record.time_us,
                                  cur_record.total_allocated + last_record.total_allocated,
                                  cur_record.total_reserved + last_record.total_reserved,
                                  cur_record.device_tag]
        else:
            pta_ge_record_list = [Constant.PTA_GE, cur_record.time_us, cur_record.total_allocated,
                                  cur_record.total_reserved, cur_record.device_tag]
        return [cur_record_list, pta_ge_record_list]

    def generate_view(self: any, output_path: str, **kwargs) -> None:
        self._add_memory_from_cann()
        self._add_pta_memory_data()
        self._add_pta_ge_record_data()
        FileManager.create_csv_file(output_path, self.memory_data, self.OPERATOR_MEMORY, self.HEADERS_OPERATOR)
        FileManager.create_csv_file(output_path, self.size_record_list + self.component_list, self.MEMORY_RECORD,
                                    self.HEADERS_RECORD)

    def _add_pta_ge_record_data(self):
        """
        ge records are to be sorted firstly and pta records are already sorted,
        then generate ge+pta records
        """
        try:
            self.ge_record_list = sorted(self.ge_record_list, key=lambda x: x.time_us)
        except Exception:
            raise RuntimeError(f"Can't sort records for cann memory record")
        ge_ptr = 0
        pta_ptr = 0
        last_ge_record = None
        last_pta_record = None
        while ge_ptr < len(self.ge_record_list) and pta_ptr < len(self.pta_record_list):
            ge_record = self.ge_record_list[ge_ptr]
            pta_record = self.pta_record_list[pta_ptr]
            if ge_record.time_us >= pta_record.time_us:
                self.size_record_list.extend(self._combine_record(last_ge_record, pta_record))
                pta_ptr += 1
                last_pta_record = pta_record
            else:
                self.size_record_list.extend(self._combine_record(last_pta_record, ge_record))
                ge_ptr += 1
                last_ge_record = ge_record
        while ge_ptr < len(self.ge_record_list):
            ge_record = self.ge_record_list[ge_ptr]
            self.size_record_list.extend(self._combine_record(last_pta_record, ge_record))
            ge_ptr += 1
        while pta_ptr < len(self.pta_record_list):
            pta_record = self.pta_record_list[pta_ptr]
            self.size_record_list.extend(self._combine_record(last_ge_record, pta_record))
            pta_ptr += 1

    def _add_device_type_for_npu(self, npu_app_memory_file_set: set):
        if npu_app_memory_file_set:
            sub_file = next(iter(npu_app_memory_file_set))
            try:
                device_id = os.path.basename(sub_file).split(".")[0].split("_")[2]
            except IndexError:
                warn(f"Can't get npu memory device id!")
                return
            device_tag = f"NPU:{device_id}"
            for record in self.size_record_list:
                if record[0] == Constant.APP:
                    record.append(device_tag)

    def split_component_ge(self, data: list):
        for row in data:
            if row.component == Constant.GE:
                self.ge_record_list.append(row)
            else:
                self.component_list.append(row)

    def _add_memory_from_cann(self):
        """
        add ge memory and app memory from cann files
        """
        npu_app_memory_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.NPU_MEMORY)
        self.size_record_list.extend(self._get_data_from_file(npu_app_memory_file_set, NpuMemoryBean))
        self._add_device_type_for_npu(npu_app_memory_file_set)
        ge_memory_record_file = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.GE_MEMORY_RECORD)
        self.split_component_ge(self._get_data_from_file(ge_memory_record_file, GeMemoryRecordBean, bean_list=True))
        ge_op_memory_file = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.GE_OPERATOR_MEMORY)
        self.memory_data.extend(self._get_data_from_file(ge_op_memory_file, GeOpMemoryBean))

    def _find_matched_torch_op_name(self, mem_start_ts: int, torch_ops: list) -> str:
        matched_torch_op_idx = self._find_torch_ops_by_binary_search(mem_start_ts, torch_ops)
        cnt = 0
        while matched_torch_op_idx >= 0 and \
                torch_ops[matched_torch_op_idx].ts + torch_ops[matched_torch_op_idx].dur < mem_start_ts:
            matched_torch_op_idx -= 1
            cnt += 1
            if cnt >= self.MAX_FIND_LAYERS or matched_torch_op_idx < 0:
                warn(f"Can't find matched torch ops for a memory record!")
                return ""
        return torch_ops[matched_torch_op_idx].name

    def _combine_memory_record(self: any, allocate_record: MemoryUseBean,
                               release_record: MemoryUseBean, torch_ops: list) -> list:
        if not allocate_record:
            return ["", release_record.alloc_size, None, release_record.time_us, None, None, None,
                    release_record.total_allocated, release_record.total_reserved, release_record.device_tag]
        torch_name = self._find_matched_torch_op_name(allocate_record.time_us, torch_ops)
        if release_record:
            return [torch_name, allocate_record.alloc_size, allocate_record.time_us, release_record.time_us,
                    release_record.time_us - allocate_record.time_us, allocate_record.total_allocated,
                    allocate_record.total_reserved, release_record.total_allocated,
                    release_record.total_reserved, allocate_record.device_tag]
        else:
            return [torch_name, allocate_record.alloc_size, allocate_record.time_us, None, None,
                    allocate_record.total_allocated, allocate_record.total_reserved, None, None,
                    allocate_record.device_tag]

    def _add_pta_memory_data(self):
        pta_memory_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.MEMORY)
        torch_op_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.TORCH_OP)
        pta_memory_dict = {}
        torch_op_dict = {}
        pta_memory_record = []
        pta_memory_data = sorted(pta_memory_data, key=lambda x: x.time_us)
        for memory_re in pta_memory_data:
            if memory_re.is_npu():
                pta_memory_dict.setdefault(memory_re.pid, []).append(memory_re)
                self.pta_record_list.append(memory_re)
        for torch_op in torch_op_data:
            torch_op_dict.setdefault(torch_op.pid, []).append(torch_op)
        for pid_key in pta_memory_dict:
            memory_records = pta_memory_dict.get(pid_key, [])
            torch_ops = torch_op_dict.get(pid_key, [])
            if not torch_ops:
                warn(f"Lack of torch ops to connect memory record, whose process id is {pid_key}")
                continue
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
                else:
                    pta_memory_record.append(self._combine_memory_record(None, memory_record, torch_ops))
        self.memory_data.extend(pta_memory_record)
