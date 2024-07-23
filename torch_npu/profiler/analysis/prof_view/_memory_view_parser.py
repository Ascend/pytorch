from warnings import warn

from torch_npu.utils._error_code import ErrCode, prof_error
from ._base_parser import BaseParser
from ._memory_prepare_parser import MemoryPrepareParser
from ..prof_common_func._path_manager import ProfilerPathManager
from ..prof_parse._fwk_file_parser import FwkFileParser
from ..prof_common_func._file_manager import FileManager
from ..prof_common_func._constant import convert_ns2us_str
from ..prof_common_func._constant import Constant, print_error_msg
from ..prof_bean._npu_mem_bean import NpuMemoryBean
from ..prof_bean._ge_op_memory_bean import GeOpMemoryBean
from ..prof_bean._ge_memory_record_bean import GeMemoryRecordBean
from ..prof_parse._cann_file_parser import CANNFileParser, CANNDataEnum

__all__ = []


class MemoryViewParser(BaseParser):
    HEADERS_OPERATOR = ["Name", "Size(KB)", "Allocation Time(us)", "Release Time(us)", "Active Release Time(us)",
                        "Duration(us)", "Active Duration(us)", "Allocation Total Allocated(MB)",
                        "Allocation Total Reserved(MB)", "Allocation Total Active(MB)", "Release Total Allocated(MB)",
                        "Release Total Reserved(MB)", "Release Total Active(MB)", "Stream Ptr", "Device Type"]
    HEADERS_RECORD = ["Component", "Timestamp(us)", "Total Allocated(MB)", "Total Reserved(MB)", "Total Active(MB)", "Stream Ptr", "Device Type"]
    OPERATOR_MEMORY = "operator_memory.csv"
    MEMORY_RECORD = "memory_record.csv"
    MAX_FIND_LAYERS = 100

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self.size_record_list = []
        self.pta_record_list = []
        self.ge_record_list = []
        self.memory_data = []
        self.component_list = []

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
            pta_ge_record_list = [Constant.PTA_GE, convert_ns2us_str(cur_record.time_ns, tail="\t"),
                                  cur_record.total_allocated + last_record.total_allocated,
                                  cur_record.total_reserved + last_record.total_reserved,
                                  cur_record.total_active + last_record.total_active,
                                  cur_record.stream_ptr if cur_record.stream_ptr else last_record.stream_ptr,
                                  cur_record.device_tag]
        else:
            pta_ge_record_list = [Constant.PTA_GE, convert_ns2us_str(cur_record.time_ns, tail="\t"),
                                  cur_record.total_allocated, cur_record.total_reserved, cur_record.total_active,
                                  cur_record.stream_ptr, cur_record.device_tag]
        return [cur_record_list, pta_ge_record_list]

    def run(self, deps_data: dict):
        try:
            self.memory_data = deps_data.get(Constant.MEMORY_PREPARE, {}).get("memory_data", [])
            self.pta_record_list = deps_data.get(Constant.MEMORY_PREPARE, {}).get("pta_record_list", [])
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate operator_memory.csv or memory_record.csv.")
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        self._init_pta_data()
        self._add_memory_from_cann()
        self._add_pta_ge_record_data()
        FileManager.create_csv_file(self._output_path, self.memory_data, self.OPERATOR_MEMORY, self.HEADERS_OPERATOR)
        FileManager.create_csv_file(self._output_path, self.size_record_list + self.component_list, self.MEMORY_RECORD,
                                    self.HEADERS_RECORD)

    def _add_pta_ge_record_data(self):
        """
        ge records are to be sorted firstly and pta records are already sorted,
        then generate ge+pta records
        """
        try:
            self.ge_record_list = sorted(self.ge_record_list, key=lambda x: x.time_ns)
        except Exception as e:
            raise RuntimeError(f"Can't sort records for cann memory record" + prof_error(ErrCode.INTERNAL)) from e
        ge_ptr = 0
        pta_ptr = 0
        last_ge_record = None
        last_pta_record = None
        while ge_ptr < len(self.ge_record_list) and pta_ptr < len(self.pta_record_list):
            ge_record = self.ge_record_list[ge_ptr]
            pta_record = self.pta_record_list[pta_ptr]
            if ge_record.time_ns >= pta_record.time_ns:
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

    def _add_device_type_for_npu(self):
        for record in self.size_record_list:
            if record[0] == Constant.APP:
                record[-1] = f"NPU:{record[-1]}"

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
        app_record_data = self._get_data_from_file(npu_app_memory_file_set, NpuMemoryBean)
        self.size_record_list.extend(app_record_data)
        self._add_device_type_for_npu()
        ge_memory_record_file = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.GE_MEMORY_RECORD)
        self.split_component_ge(self._get_data_from_file(ge_memory_record_file, GeMemoryRecordBean, bean_list=True))
        ge_op_memory_file = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.GE_OPERATOR_MEMORY)
        self.memory_data.extend(self._get_data_from_file(ge_op_memory_file, GeOpMemoryBean))

    def _init_pta_data(self):
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            torch_nop_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(only_fwk=True)
            deps_data = {Constant.TREE_BUILD_PARSER: torch_nop_node}
            _, pta_data = MemoryPrepareParser(Constant.MEMORY_PREPARE, self._param_dict).run(deps_data)
            self.memory_data = pta_data.get("memory_data", [])
            self.pta_record_list = pta_data.get("pta_record_list", [])
