import ast
import json
import os
import re
from enum import Enum
from json import JSONDecodeError

from ....utils.path_manager import PathManager
from ..prof_bean.event_bean import EventBean
from ..prof_common_func.constant import Constant, print_warn_msg
from ..prof_common_func.constant import convert_us2ns
from ..prof_common_func.path_manager import ProfilerPathManager
from ..prof_common_func.file_manager import FileManager


class CANNDataEnum(Enum):
    OP_SUMMARY = 0
    NPU_MEMORY = 1
    MSPROF_TIMELINE = 2
    STEP_TRACE = 3
    GE_MEMORY_RECORD = 4
    GE_OPERATOR_MEMORY = 5
    L2_CACHE = 6
    AI_CPU = 7
    COMMUNICATION = 8
    MATRIX = 9
    OP_STATISTIC = 10
    NPU_MODULE_MEM = 11


class CANNFileParser:
    START_FLOW = "s"
    END_FLOW = "f"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    ANALYZE = "analyze"
    HOST_TO_DEVICE = "HostToDevice"
    CANN_DATA_MATCH = {
        CANNDataEnum.OP_SUMMARY: [r"^op_summary_\d+_\d+\.csv", r"^op_summary_\d+_\d+_\d+\.csv",
                                  r"^op_summary_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.NPU_MEMORY: [r"^npu_mem_\d+_\d+\.csv", r"^npu_mem_\d+_\d+_\d+\.csv",
                                  r"^npu_mem_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.MSPROF_TIMELINE: [r"^msprof_\d+_\d+\.json", r"^msprof_\d+_\d+_\d+\.json",
                                       r"^msprof_\d+_\d+_\d+_\d+\.json", r"^msprof_\d+_\d+_slice_\d+\.json",
                                       r"^msprof_\d+_\d+_slice_\d+_\d+\.json",
                                       r"^msprof_\d+_\d+_\d+_slice_\d+\.json",
                                       r"^msprof_\d+_\d+_\d+_slice_\d+_\d+\.json"],
        CANNDataEnum.STEP_TRACE: [r"^step_trace_\d+_\d+\.csv", r"^step_trace_\d+_\d+_\d+\.csv",
                                  r"^step_trace_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.GE_MEMORY_RECORD: [r"^ge_memory_record_\d+_\d+\.csv", r"^ge_memory_record_\d+_\d+_\d+\.csv",
                                        r"^ge_memory_record_\d+_\d+_\d+_\d+\.csv", r"^memory_record_\d+_\d+\.csv",
                                        r"^memory_record_\d+_\d+_\d+\.csv", r"^memory_record_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.GE_OPERATOR_MEMORY: [r"^ge_operator_memory_\d+_\d+\.csv", r"^ge_operator_memory_\d+_\d+_\d+\.csv",
                                          r"^ge_operator_memory_\d+_\d+_\d+_\d+\.csv", r"^operator_memory_\d+_\d+\.csv",
                                          r"^operator_memory_\d+_\d+_\d+\.csv",
                                          r"^operator_memory_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.L2_CACHE: [r"^l2_cache_\d+_\d+\.csv", r"^l2_cache_\d+_\d+_\d+\.csv",
                                r"^l2_cache_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.AI_CPU: [r"^aicpu_\d+_\d+\.csv", r"^aicpu_\d+_\d+_\d+\.csv", r"^aicpu_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.COMMUNICATION: [r"^communication\.json"],
        CANNDataEnum.MATRIX: [r"^communication_matrix\.json"],
        CANNDataEnum.OP_STATISTIC: [r"^op_statistic_\d+_\d+\.csv", r"^op_statistic_\d+_\d+_\d+\.csv",
                                    r"^op_statistic_\d+_\d+_\d+_\d+\.csv"],
        CANNDataEnum.NPU_MODULE_MEM: [r"^npu_module_mem_\d_\d_\d+\.csv"],
    }

    def __init__(self, profiler_path: str):
        self._cann_path = ProfilerPathManager.get_cann_path(profiler_path)
        self._file_dict = {}
        self._file_dispatch()

    @classmethod
    def _json_load(cls, data: str) -> list:
        if not data:
            return []
        try:
            data = json.loads(data)
        except JSONDecodeError as e:
            raise RuntimeError("Invalid CANN trace data.") from e
        if not isinstance(data, list):
            return []
        if data and not isinstance(data[0], dict):
            return []
        return data

    @classmethod
    def _json_dict_load(cls, data: str) -> dict:
        if not data:
            return {}
        try:
            data = json.loads(data)
        except JSONDecodeError as e:
            raise RuntimeError("Invalid communication data.") from e
        if not isinstance(data, dict):
            return {}
        return data

    @classmethod
    def combine_acl_to_npu(cls, timeline_data: list) -> dict:
        flow_dict, event_dict = {}, {}
        for data in timeline_data:
            if data.get("cat") == cls.HOST_TO_DEVICE and data.get("ph") == cls.START_FLOW:
                flow_dict.setdefault(data.get("id", 0), {}).setdefault("start", data)
                continue
            if data.get("cat") == cls.HOST_TO_DEVICE and data.get("ph") == cls.END_FLOW:
                flow_dict.setdefault(data.get("id", 0), {}).setdefault("end", data)
                continue
            if data.get("ph") == "X":
                pid = data.get("pid")
                tid = data.get("tid")
                ts = data.get("ts")
                unique_id = f"{pid}-{tid}-{ts}"
                event_dict[unique_id] = data
        acl_to_npu_dict = {}
        for flow in flow_dict.values():
            start_event = flow.get("start")
            end_event = flow.get("end")
            if start_event and end_event:
                pid = end_event.get("pid")
                tid = end_event.get("tid")
                ts = end_event.get("ts")
                unique_id = f"{pid}-{tid}-{ts}"
                kernel_event = event_dict.get(unique_id)
                if not kernel_event:
                    continue
                acl_to_npu_dict.setdefault(convert_us2ns(start_event.get("ts", 0)), []).append(EventBean(kernel_event))
        return acl_to_npu_dict

    def get_timeline_all_data(self) -> list:
        timeline_data = []
        msprof_file_list = self._file_dict.get(CANNDataEnum.MSPROF_TIMELINE, set())
        for msprof_file in msprof_file_list:
            data = self._json_load(FileManager.file_read_all(msprof_file, "rt"))
            timeline_data.extend(data)
        return timeline_data

    def get_analyze_communication_data(self, file_type: Enum) -> dict:
        communication_data = {}
        communication_file_set = self._file_dict.get(file_type, set())
        if communication_file_set:
            # only need to read one file if there exist more than one files
            sub_file = next(iter(communication_file_set))
            communication_data = self._json_dict_load(FileManager.file_read_all(sub_file, "rt"))
        return communication_data

    def get_acl_to_npu_data(self):
        return self.combine_acl_to_npu(self.get_timeline_all_data())

    def get_file_list_by_type(self, file_type: CANNDataEnum) -> set:
        return self._file_dict.get(file_type, set())

    def get_localtime_diff(self) -> float:
        localtime_diff = 0
        if not self._cann_path:
            return localtime_diff
        start_info_path = ProfilerPathManager.get_start_info_path(self._cann_path)
        if not start_info_path:
            return localtime_diff
        try:
            info_json = ast.literal_eval(FileManager.file_read_all(start_info_path, "rt"))
            localtime_diff = convert_us2ns(info_json.get(Constant.CANN_BEGIN_TIME, 0)) - int(
                info_json.get(Constant.CANN_BEGIN_MONOTONIC, 0))
        except Exception:
            print_warn_msg("Failed to get CANN localtime diff.")
        return localtime_diff

    def del_summary_and_timeline_data(self):
        device_path = ProfilerPathManager.get_device_path(self._cann_path)
        if not device_path:
            return
        summary_path = os.path.join(device_path, "summary")
        timeline_path = os.path.join(device_path, "timeline")
        PathManager.remove_path_safety(summary_path)
        PathManager.remove_path_safety(timeline_path)

    def _file_dispatch(self):
        all_file_list = ProfilerPathManager.get_device_all_file_list_by_type(self._cann_path, self.SUMMARY)
        all_file_list += ProfilerPathManager.get_device_all_file_list_by_type(self._cann_path, self.TIMELINE)
        all_file_list += ProfilerPathManager.get_analyze_all_file(self._cann_path, self.ANALYZE)
        for file_path in all_file_list:
            if not os.path.isfile(file_path):
                continue
            for data_type, re_match_exp_list in self.CANN_DATA_MATCH.items():
                for re_match_exp in re_match_exp_list:
                    if re.match(re_match_exp, os.path.basename(file_path)):
                        self._file_dict.setdefault(data_type, set()).add(file_path)
