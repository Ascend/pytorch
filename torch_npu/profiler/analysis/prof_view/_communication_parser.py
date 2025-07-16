import re
from collections import defaultdict

from ._base_parser import BaseParser
from ..prof_bean._torch_op_node import TorchOpNode
from ..prof_common_func._constant import Constant, print_warn_msg
from ..prof_common_func._file_manager import FileManager
from ..prof_parse._cann_file_parser import CANNFileParser
from ..prof_parse._cann_file_parser import CANNDataEnum
from ..prof_common_func._constant import convert_us2ns
from ..prof_common_func._log import ProfilerLogger
from ..prof_parse._fwk_cann_relation_parser import FwkCANNRelationParser

__all__ = []


class CommunicationParser(BaseParser):
    """
    load and split communication info by step
    """
    COMMUNICATION_TIME_INFO = "Communication Time Info"
    START_TIMESTAMP = "Start Timestamp(us)"
    COMMUNICATION_BANDWIDTH_INFO = "Communication Bandwidth Info"
    HCOM_SEND = "hcom_send"
    HCOM_RECEIVE = "hcom_receive"
    HCOM_BATCHSENDRECV = "hcom_batchsendrecv"
    HCCL_PATTERN = r"send|reduce|invalid|broadcast|allreduce|" \
                   "receive|allgather|reducescatter|scatter|alltoall|alltoallv|alltoallvc|batchsendrecv"
    TOTAL = "total"
    SYNCHRONIZATION_TIME_RATIO = "Synchronization Time Ratio"
    SYNCHRONIZATION_TIME_MS = "Synchronization Time(ms)"
    WAIT_TIME_RATIO = "Wait Time Ratio"
    TRANSIT_TIME_MS = "Transit Time(ms)"
    TRANSIT_SIZE_MB = "Transit Size(MB)"
    SIZE_DISTRIBUTION = "Size Distribution"
    WAIT_TIME_MS = "Wait Time(ms)"
    BANDWIDTH_GB_S = "Bandwidth(GB/s)"
    COMMUNICATION = "communication.json"
    COMMUNICATION_MATRIX = "communication_matrix.json"
    P2P = "p2p"
    COLLECTIVE = "collective"
    TRANSPORT_TYPE = "Transport Type"

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._root_node = TorchOpNode()
        self._kernel_dict = {}
        self.step_list = []

    @staticmethod
    def combine_size_distribution(op_dict: dict, total_dict: dict):
        for size, size_info in op_dict.items():
            total_dict[size][0] += size_info[0]
            total_dict[size][1] += size_info[1]

    @staticmethod
    def compute_ratio(dividend: float, divisor: float):
        if abs(divisor) < 1e-15:
            return 0
        else:
            return round(dividend / divisor, 4)

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "CommunicationParser")
        self.logger = ProfilerLogger.get_instance()
        try:
            self._init_step_list(deps_data)
            self.generate_view()
        except Exception as e:
            self.logger.error("Failed to generate communication.json or communication_matrix.json, error: %s", str(e), exc_info=True)
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        self.generate_communication(self._output_path)
        self.generate_matrix(self._output_path)

    def generate_communication(self, output_path: str):
        communication_data = CANNFileParser(self._profiler_path).get_analyze_communication_data(
            CANNDataEnum.COMMUNICATION)
        if not communication_data:
            return
        self.split_comm_op_by_step(communication_data)
        output_communication = {}
        for step_info in self.step_list:
            step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
            output_communication[step] = self.get_communication_ops_dict(step_info.get("comm_ops"))
        FileManager.create_json_file(output_path, output_communication, self.COMMUNICATION)

    def generate_matrix(self, output_path: str):
        matrix_data = CANNFileParser(self._profiler_path).get_analyze_communication_data(CANNDataEnum.MATRIX)
        if not matrix_data:
            return
        matrix_data_by_step = self.split_matrix_by_step(matrix_data)
        output_matrix_data = {}
        for step, comm_matrix_data in matrix_data_by_step.items():
            output_matrix_data[step] = self.get_matrix_ops_dict(comm_matrix_data)
        FileManager.create_json_file(output_path, output_matrix_data, self.COMMUNICATION_MATRIX)

    def split_comm_op_by_step(self, communication_data: dict):
        if len(self.step_list) == 1:
            self.step_list[0]["comm_ops"] = communication_data
        for communication_op, communication_op_info in communication_data.items():
            start_time = communication_op_info.get(self.COMMUNICATION_TIME_INFO, {}).get(self.START_TIMESTAMP)
            start_time = convert_us2ns(start_time)
            for step_info in self.step_list:
                if step_info.get("start_ts", -1) <= start_time <= step_info.get("end_ts", -1):
                    step_info.get("comm_ops", {})[communication_op] = communication_op_info
                    break

    def split_communication_p2p_ops(self, op_data: dict):
        comm_op_dict = {self.P2P: {}, self.COLLECTIVE: {}}
        for communication_op, communication_info in op_data.items():
            lower_op_name = communication_op.lower()
            if lower_op_name.startswith(self.HCOM_SEND) or lower_op_name.startswith(self.HCOM_RECEIVE) \
                    or lower_op_name.startswith(self.HCOM_BATCHSENDRECV):
                comm_op_dict[self.P2P][communication_op] = communication_info
            elif lower_op_name.startswith(self.TOTAL):
                continue
            else:
                comm_op_dict[self.COLLECTIVE][communication_op] = communication_info
        return comm_op_dict

    def compute_total_link_info(self, op_matrix_data: dict):
        total_op_info = defaultdict(lambda: {
            self.TRANSPORT_TYPE: '',
            self.TRANSIT_TIME_MS: 0,
            self.TRANSIT_SIZE_MB: 0
        })
        for op_name, link_dict in op_matrix_data.items():
            for link, link_info in link_dict.items():
                total_op_info[link][self.TRANSIT_TIME_MS] += link_info.get(self.TRANSIT_TIME_MS, 0)
                total_op_info[link][self.TRANSIT_SIZE_MB] += link_info.get(self.TRANSIT_SIZE_MB, 0)
                total_op_info[link][self.TRANSPORT_TYPE] = link_info.get(self.TRANSPORT_TYPE, 0)

        for link, link_info_dict in total_op_info.items():
            total_op_info[link][self.BANDWIDTH_GB_S] = \
                self.compute_ratio(total_op_info[link].get(self.TRANSIT_SIZE_MB, 0),
                                   total_op_info[link].get(self.TRANSIT_TIME_MS))
        op_matrix_data['Total Op Info'] = total_op_info

    def split_matrix_by_step(self, matrix_data: dict) -> dict:
        matrix_data_by_step = {}
        if self.is_step_list_empty():
            matrix_data_by_step["step"] = matrix_data
            return matrix_data_by_step

        for comm_op in matrix_data:
            for step_info in self.step_list:
                if comm_op in step_info.get("comm_ops", {}):
                    step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
                    matrix_data_by_step.setdefault(step, {})[comm_op] = matrix_data.get(comm_op)
                    break
        return matrix_data_by_step

    def get_communication_ops_dict(self, op_data: dict) -> dict:
        comm_op_dict = self.split_communication_p2p_ops(op_data)
        self.compute_total_info(comm_op_dict[self.P2P])
        self.compute_total_info(comm_op_dict[self.COLLECTIVE])
        return comm_op_dict

    def get_matrix_ops_dict(self, op_data: dict) -> dict:
        comm_op_dict = self.split_communication_p2p_ops(op_data)
        comm_op_dict[self.P2P] = self.integrate_matrix_data(self.get_comm_type(comm_op_dict[self.P2P]))
        comm_op_dict[self.COLLECTIVE] = self.integrate_matrix_data(self.get_comm_type(comm_op_dict[self.COLLECTIVE]))
        return comm_op_dict

    def get_comm_type(self, op_data: dict) -> dict:
        new_comm_op_dict = defaultdict(list)
        for communication_op, communication_info in op_data.items():
            match_obj = re.compile(self.HCCL_PATTERN).search((communication_op.lower()))
            if match_obj:
                comm_op_type = match_obj.group()
            else:
                comm_op_type = communication_op.split("__")[0]
                print_warn_msg(f"Unknown communication op type: {comm_op_type}")
            for link, data in communication_info.items():
                new_comm_op_name = (comm_op_type, communication_op.split("@")[-1], link)
                data['Op Name'] = communication_op.split("@")[0]
                new_comm_op_dict[new_comm_op_name].append(data)
        return new_comm_op_dict

    def integrate_matrix_data(self, new_comm_op_dict: dict):
        """integrate the matrix data"""
        comm_op_dict = defaultdict(dict)
        for new_comm_op_name, data in new_comm_op_dict.items():
            data.sort(key=lambda x: x[self.BANDWIDTH_GB_S], reverse=True)
            t_type = data[0].get(self.TRANSPORT_TYPE, '')
            t_size = sum(x.get(self.TRANSIT_SIZE_MB, 0) for x in data)
            t_time = sum(x.get(self.TRANSIT_TIME_MS, 0) for x in data)
            bandwidth = self.compute_ratio(t_size, t_time)

            link = new_comm_op_name[2]
            new_comm_op_name_top1 = f'{new_comm_op_name[0]}-top1@{new_comm_op_name[1]}'
            new_comm_op_name_middle = f'{new_comm_op_name[0]}-middle@{new_comm_op_name[1]}'
            new_comm_op_name_bottom1 = f'{new_comm_op_name[0]}-bottom1@{new_comm_op_name[1]}'
            new_comm_op_name_bottom2 = f'{new_comm_op_name[0]}-bottom2@{new_comm_op_name[1]}'
            new_comm_op_name_bottom3 = f'{new_comm_op_name[0]}-bottom3@{new_comm_op_name[1]}'
            new_comm_op_name_total = f'{new_comm_op_name[0]}-total@{new_comm_op_name[1]}'
            comm_op_dict[new_comm_op_name_top1].update({link: data[0]})
            comm_op_dict[new_comm_op_name_middle].update({link: data[len(data) // 2]})
            comm_op_dict[new_comm_op_name_bottom1].update({link: data[-1]})
            comm_op_dict[new_comm_op_name_total].update({link: {
                self.TRANSPORT_TYPE: t_type,
                self.TRANSIT_SIZE_MB: t_size,
                self.TRANSIT_TIME_MS: t_time,
                self.BANDWIDTH_GB_S: bandwidth
            }})
            if len(data) >= 2:
                comm_op_dict[new_comm_op_name_bottom2].update({link: data[-2]})
            if len(data) >= 3:
                comm_op_dict[new_comm_op_name_bottom3].update({link: data[-3]})
        return comm_op_dict

    def is_step_list_empty(self):
        for step_info in self.step_list:
            if step_info.get("comm_ops"):
                return False
        return True

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
        total_time_info_dict[self.WAIT_TIME_RATIO] = \
            self.compute_ratio(total_time_info_dict.get(self.WAIT_TIME_MS, 0),
                               total_time_info_dict.get(self.WAIT_TIME_MS, 0) +
                               total_time_info_dict.get(self.TRANSIT_TIME_MS, 0))
        total_time_info_dict[self.SYNCHRONIZATION_TIME_RATIO] = \
            self.compute_ratio(total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0),
                               total_time_info_dict.get(self.TRANSIT_TIME_MS, 0) +
                               total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0))

    def compute_bandwidth_ratio(self, total_bandwidth_info_dict: dict):
        for transport_type, bandwidth_dict in total_bandwidth_info_dict.items():
            self.compute_ratio(bandwidth_dict.get(self.TRANSIT_SIZE_MB, 0), bandwidth_dict.get(self.TRANSIT_TIME_MS, 0))

    def _init_step_list(self, deps_data: dict):
        torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
        if torch_op_node:
            kernels_dict = deps_data.get(Constant.RELATION_PARSER, {})
            if not kernels_dict:
                self.logger.error("Init step list failed, the kernel dict is empty.")
            self.step_list = FwkCANNRelationParser(self._profiler_path).get_step_range(torch_op_node[0], kernels_dict)

        if not self.step_list:
            self.step_list = [{"step_id": None, "start_ts": 0, "end_ts": float('inf'), "comm_ops": {}}]
