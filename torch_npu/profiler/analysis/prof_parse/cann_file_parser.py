import json
import os
import re
import subprocess
from enum import Enum
from json import JSONDecodeError

from ..prof_common_func.constant import Constant
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.path_manager import PathManager
from ..prof_bean.step_trace_bean import StepTraceBean


class CANNDataEnum(Enum):
    OP_SUMMARY = 0
    NPU_MEMORY = 1
    MSPROF_TIMELINE = 2
    STEP_TRACE = 3


class CANNFileParser:
    COMMAND_SUCCESS = 0
    ACL_TO_NPU = "acl_to_npu"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    CANN_DATA_MATCH = {
        CANNDataEnum.OP_SUMMARY: [r"^op_summary_\d_\d+\.csv", r"^op_summary_\d_\d+_\d+\.csv"],
        CANNDataEnum.NPU_MEMORY: [r"^npu_mem_\d_\d+\.csv", r"^npu_mem_\d_\d+_\d+\.csv"],
        CANNDataEnum.MSPROF_TIMELINE: [r"^msprof_\d_\d+\.json", r"^msprof_\d_\d+_\d+\.json"],
        CANNDataEnum.STEP_TRACE: [r"^step_trace_\d_\d+\.csv", r"^step_trace_\d_\d+_\d+\.csv"]
    }

    def __init__(self, profiler_path: str):
        self._cann_path = PathManager.get_cann_path(profiler_path)
        self._file_dict = {}
        self._file_dispatch()

    @classmethod
    def _json_load(cls, data: str) -> list:
        if not data:
            return []
        try:
            data = json.loads(data)
        except JSONDecodeError:
            raise RuntimeError("Invalid CANN trace data.")
        if not isinstance(data, list):
            return []
        if data and not isinstance(data[0], dict):
            return []
        if data and "ph" not in data[0].keys():
            return []
        return data

    def export_cann_profiling(self):
        if not os.path.isdir(self._cann_path):
            return
        completed_process = subprocess.run(["msprof", "--export=on", f"--output={self._cann_path}"],
                                           capture_output=True, timeout=2400)
        if completed_process.returncode != self.COMMAND_SUCCESS:
            raise RuntimeError("Export CANN Profiling data failed, please verify that the "
                               "ascend-toolkit is installed and set-env.sh is sourced.")
        self._file_dispatch()
        step_trace_file_set = self.get_file_list_by_type(CANNDataEnum.STEP_TRACE)
        if not step_trace_file_set:
            return
        step_file = step_trace_file_set.pop()
        step_trace_data = FileManager.read_csv_file(step_file, StepTraceBean)
        parsed_step = os.path.basename(step_file).split(".")[0].split("_")[-1]
        for data in step_trace_data:
            step_id = data.step_id
            if step_id != Constant.INVALID_VALUE and step_id != parsed_step:
                completed_process = subprocess.run(
                    ["msprof", "--export=on", f"--output={self._cann_path}", f"--iteration-id={step_id}"],
                    capture_output=True, timeout=2400)
                if completed_process.returncode != self.COMMAND_SUCCESS:
                    raise RuntimeError("Export CANN Profiling data failed, please verify that the "
                                       "ascend-toolkit is installed and set-env.sh is sourced.")

    def get_timeline_all_data(self) -> list:
        timeline_data = []
        msprof_file_list = self._file_dict.get(CANNDataEnum.MSPROF_TIMELINE, set())
        for msprof_file in msprof_file_list:
            data = self._json_load(FileManager.file_read_all(msprof_file, "rt"))
            timeline_data.extend(data)
        return timeline_data

    def get_acl_and_npu_data(self) -> dict:
        acl_and_npu_data = {}
        all_data = self.get_timeline_all_data()
        for data in all_data:
            if data.get("name", "") in Constant.ACL_OP_EXE_NAME:
                acl_and_npu_data.setdefault(data.get("ts"), [])
            elif data.get("name") == self.ACL_TO_NPU and data.get("ph") == "s":
                acl_and_npu_data.setdefault(data.get("ts"), []).append(data.get("id"))
        return acl_and_npu_data

    def get_file_list_by_type(self, file_type: CANNDataEnum) -> set:
        return self._file_dict.get(file_type, set())

    def _file_dispatch(self):
        all_file_list = PathManager.get_device_all_file_list_by_type(self._cann_path, self.SUMMARY)
        all_file_list += PathManager.get_device_all_file_list_by_type(self._cann_path, self.TIMELINE)
        for file_path in all_file_list:
            if not os.path.isfile(file_path):
                continue
            for data_type, re_match_exp_list in self.CANN_DATA_MATCH.items():
                for re_match_exp in re_match_exp_list:
                    if re.match(re_match_exp, os.path.basename(file_path)):
                        self._file_dict.setdefault(data_type, set()).add(file_path)
