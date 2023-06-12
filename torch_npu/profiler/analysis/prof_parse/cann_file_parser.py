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

import json
import os
import re
import subprocess
from enum import Enum
from json import JSONDecodeError

from ..prof_bean.event_bean import EventBean
from ..prof_common_func.constant import Constant
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.path_manager import PathManager
from ..prof_bean.step_trace_bean import StepTraceBean


class CANNDataEnum(Enum):
    OP_SUMMARY = 0
    NPU_MEMORY = 1
    MSPROF_TIMELINE = 2
    STEP_TRACE = 3
    GE_MEMORY_RECORD = 4
    GE_OPERATOR_MEMORY = 5
    L2_CACHE = 6
    AI_CPU = 7


class CANNFileParser:
    COMMAND_SUCCESS = 0
    ACL_TO_NPU = "acl_to_npu"
    START_FLOW = "s"
    END_FLOW = "f"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    CANN_DATA_MATCH = {
        CANNDataEnum.OP_SUMMARY: [r"^op_summary_\d_\d+\.csv", r"^op_summary_\d_\d+_\d+\.csv"],
        CANNDataEnum.NPU_MEMORY: [r"^npu_mem_\d_\d+\.csv", r"^npu_mem_\d_\d+_\d+\.csv"],
        CANNDataEnum.MSPROF_TIMELINE: [r"^msprof_\d_\d+\.json", r"^msprof_\d_\d+_\d+\.json"],
        CANNDataEnum.STEP_TRACE: [r"^step_trace_\d_\d+\.csv", r"^step_trace_\d_\d+_\d+\.csv"],
        CANNDataEnum.GE_MEMORY_RECORD: [r"^ge_memory_record_\d_\d+\.csv", r"^ge_memory_record_\d_\d+_\d+\.csv"],
        CANNDataEnum.GE_OPERATOR_MEMORY: [r"^ge_operator_memory_\d_\d+\.csv", r"^ge_operator_memory_\d_\d+_\d+\.csv"],
        CANNDataEnum.L2_CACHE: [r"^l2_cache_\d_\d+\.csv", r"^l2_cache_\d_\d+_\d+\.csv"],
        CANNDataEnum.AI_CPU: [r"^aicpu_\d_\d+\.csv", r"^aicpu_\d_\d+_\d+\.csv"]
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

    def get_acl_to_npu_data(self) -> dict:
        flow_start_dict, flow_end_dict = {}, {}
        all_data = self.get_timeline_all_data()
        for data in all_data:
            event_bean = EventBean(data)
            if event_bean.is_flow_start_event():
                flow_start_dict[event_bean.id] = event_bean.ts
            elif event_bean.is_flow_end_event():
                flow_end_dict[event_bean.unique_id] = event_bean.id
        acl_to_npu_dict = {}
        for data in all_data:
            event_bean = EventBean(data)
            if event_bean.is_x_event():
                corr_id = flow_end_dict.get(event_bean.unique_id)
                acl_ts = flow_start_dict.get(corr_id)
                if corr_id is not None and acl_ts is not None:
                    acl_to_npu_dict.setdefault(acl_ts, []).append(event_bean)
        return acl_to_npu_dict

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
