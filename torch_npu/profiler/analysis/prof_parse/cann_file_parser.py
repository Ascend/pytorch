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


class CANNFileParser:
    ACL_TO_NPU = "acl_to_npu"
    START_FLOW = "s"
    END_FLOW = "f"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    ANALYZE = "analyze"
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
        flow_start_dict, flow_end_dict = {}, {}
        for data in timeline_data:
            event_bean = EventBean(data)
            if event_bean.is_flow_start_event():
                flow_start_dict[event_bean.id] = event_bean.ts
            elif event_bean.is_flow_end_event():
                flow_end_dict[event_bean.unique_id] = event_bean.id
        acl_to_npu_dict = {}
        for data in timeline_data:
            event_bean = EventBean(data)
            if event_bean.is_x_event():
                corr_id = flow_end_dict.get(event_bean.unique_id)
                acl_ts = flow_start_dict.get(corr_id)
                if corr_id is not None and acl_ts is not None:
                    acl_to_npu_dict.setdefault(acl_ts, []).append(event_bean)
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
