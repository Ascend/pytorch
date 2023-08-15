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

from ..prof_common_func.file_manager import FileManager
from ..prof_view.base_view_parser import BaseViewParser
from ..prof_parse.cann_file_parser import CANNFileParser
from ..level_config import LevelConfig
from ..prof_common_func.global_var import GlobalVar
from collections import defaultdict


class CommunicationParser(BaseViewParser):
    """
    load and split communication info by step
    """
    COMMUNICATION_TIME_INFO = "Communication Time Info"
    START_TIMESTAMP = "Start Timestamp(us)"
    COMMUNICATION_BANDWIDTH_INFO = "Communication Bandwidth Info"
    HCOM_SEND = "hcom_send"
    HCOM_RECEIVE = "hcom_receive"
    TOTAL = "Total"
    SYNCHRONIZATION_TIME_RATIO = "Synchronization Time Ratio"
    SYNCHRONIZATION_TIME_MS = "Synchronization Time(ms)"
    WAIT_TIME_RATIO = "Wait Time Ratio"
    TRANSIT_TIME_MS = "Transit Time(ms)"
    TRANSIT_SIZE_MB = "Transit Size(MB)"
    SIZE_DISTRIBUTION = "Size Distribution"
    WAIT_TIME_MS = "Wait Time(ms)"
    BANDWIDTH_GB_S = "Bandwidth(GB/s)"
    COMMUNICATION = "communication.json"
    P2P = "p2p"
    COLLECTIVE = "collective"

    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    @staticmethod
    def combine_size_distribution(op_dict: dict, total_dict: dict):
        for size, size_info in op_dict.items():
            total_dict[size][0] += size_info[0]
            total_dict[size][1] += size_info[1]

    @staticmethod
    def compute_ratio(dividend: float, divisor: float):
        if divisor == 0:
            return 0
        else:
            return round(dividend / divisor, 4)

    def generate_view(self, output_path: str = None) -> None:
        communication_data = CANNFileParser(self._profiler_path).get_analyze_communication_data()
        if not communication_data:
            return
        step_list = self.split_comm_op_by_step(communication_data)
        output_communication = {}
        for step_info in step_list:
            step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
            output_communication[step] = self.split_communication_ops(step_info)
        FileManager.create_json_file(self._profiler_path, output_communication, self.COMMUNICATION)

    def split_comm_op_by_step(self, communication_data: dict) -> list:
        step_list = GlobalVar.get_step_id_list()
        if len(step_list) == 1:
            step_list[0]["comm_ops"] = communication_data
        for communication_op, communication_op_info in communication_data.items():
            start_time = communication_op_info.get(self.COMMUNICATION_TIME_INFO, {}).get(self.START_TIMESTAMP)
            for step_info in step_list:
                if step_info["start_ts"] <= start_time <= step_info["end_ts"]:
                    step_info["comm_ops"][communication_op] = communication_op_info
        return step_list

    def split_communication_ops(self, step_info: dict) -> dict:
        comm_op_dict = {self.P2P: {}, self.COLLECTIVE: {}}
        for communication_op, communication_op_info in step_info["comm_ops"].items():
            if communication_op.startswith(self.HCOM_SEND) or communication_op.startswith(self.HCOM_RECEIVE):
                comm_op_dict[self.P2P][communication_op] = communication_op_info
            elif communication_op.startswith(self.TOTAL):
                continue
            else:
                comm_op_dict[self.COLLECTIVE][communication_op] = communication_op_info
        self.compute_total_info(comm_op_dict[self.P2P])
        self.compute_total_info(comm_op_dict[self.COLLECTIVE])
        return comm_op_dict

    def compute_total_info(self, comm_ops: dict):
        if not comm_ops:
            return
        total_time_info_dict = defaultdict(float)
        total_bandwidth_info_dict = {}
        for communication_op, communication_op_info in comm_ops.items():
            for com_info, com_info_dict in communication_op_info.items():
                if com_info == self.COMMUNICATION_TIME_INFO:
                    self.combine_time_info(com_info_dict, total_time_info_dict)
                if com_info == self.COMMUNICATION_BANDWIDTH_INFO:
                    self.combine_bandwidth_info(com_info_dict, total_bandwidth_info_dict)
        self.compute_time_ratio(total_time_info_dict)
        self.compute_bandwidth_ratio(total_bandwidth_info_dict)
        comm_ops['Total Op Info'] = {
            self.COMMUNICATION_TIME_INFO: total_time_info_dict,
            self.COMMUNICATION_BANDWIDTH_INFO: total_bandwidth_info_dict
        }

    def combine_time_info(self, com_info_dict: dict, total_time_info_dict: dict):
        ratio_list = [self.WAIT_TIME_RATIO, self.SYNCHRONIZATION_TIME_RATIO]
        for time_info in com_info_dict:
            if time_info not in ratio_list and time_info != self.START_TIMESTAMP:
                total_time_info_dict[time_info] += com_info_dict.get(time_info)

    def combine_bandwidth_info(self, com_info_dict: dict, total_bandwidth_info_dict: dict):
        add_list = [self.TRANSIT_TIME_MS, self.TRANSIT_SIZE_MB]
        dict_list = [self.SIZE_DISTRIBUTION]
        for transport_type, part_transport_dict in com_info_dict.items():
            if transport_type not in total_bandwidth_info_dict:
                total_bandwidth_info_dict[transport_type] = {
                    self.TRANSIT_TIME_MS: 0,
                    self.TRANSIT_SIZE_MB: 0,
                    self.SIZE_DISTRIBUTION: defaultdict(lambda: [0, 0])
                }
            for bandwidth_msg, value in part_transport_dict.items():
                if bandwidth_msg in add_list:
                    total_bandwidth_info_dict[transport_type][bandwidth_msg] += value
                if bandwidth_msg in dict_list:
                    self.combine_size_distribution(value, total_bandwidth_info_dict[transport_type][bandwidth_msg])

    def compute_time_ratio(self, total_time_info_dict: dict):
        total_time_info_dict[self.WAIT_TIME_RATIO] =\
            self.compute_ratio(total_time_info_dict.get(self.WAIT_TIME_MS, 0),
                               total_time_info_dict.get(self.WAIT_TIME_MS, 0) +
                               total_time_info_dict.get(self.TRANSIT_TIME_MS, 0))
        total_time_info_dict[self.SYNCHRONIZATION_TIME_RATIO] = \
            self.compute_ratio(total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0),
                               total_time_info_dict.get(self.TRANSIT_TIME_MS, 0) +
                               total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0))

    def compute_bandwidth_ratio(self, total_bandwidth_info_dict: dict):
        for transport_type, bandwidth_dict in total_bandwidth_info_dict.items():
            if bandwidth_dict.get(self.TRANSIT_TIME_MS) == 0:
                bandwidth_dict[self.BANDWIDTH_GB_S] = 0
            else:
                bandwidth_dict[self.BANDWIDTH_GB_S] = \
                    self.compute_ratio(bandwidth_dict.get(self.TRANSIT_SIZE_MB, 0),
                                       bandwidth_dict.get(self.TRANSIT_TIME_MS, 0))
