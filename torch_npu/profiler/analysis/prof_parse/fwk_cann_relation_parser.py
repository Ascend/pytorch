from .fwk_file_parser import FwkFileParser
from ..prof_bean.torch_op_node import TorchOpNode
from ..prof_common_func.constant import Constant, print_error_msg
from ..prof_parse.cann_file_parser import CANNFileParser


class FwkCANNRelationParser:
    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    @classmethod
    def combine_kernel_dict(cls, acl_to_npu_dict: dict, dequeue_data_list: list):
        if not dequeue_data_list:
            return acl_to_npu_dict
        kernel_dict = {}
        index = 0
        acl_start_time_list = sorted(list(acl_to_npu_dict.keys()))
        for acl_start_time in acl_start_time_list:
            while index < len(dequeue_data_list):
                if dequeue_data_list[index].ts > acl_start_time:
                    break
                if acl_start_time <= dequeue_data_list[index].ts + dequeue_data_list[index].dur:
                    kernel_dict.setdefault(dequeue_data_list[index].corr_id, []).extend(
                        acl_to_npu_dict.get(acl_start_time, []))
                    break
                index += 1
        return kernel_dict

    @classmethod
    def _update_step_node_info(cls, step_node_list: list, acl_start_time_list: list):
        step_node_list.sort(key=lambda x: x.start_time)
        index = 0
        for acl_start_time in acl_start_time_list:
            while index < len(step_node_list):
                step_node = step_node_list[index]
                if step_node.start_time <= acl_start_time <= step_node.end_time:
                    step_node.update_corr_id_total(acl_start_time)
                    break
                if acl_start_time < step_node.start_time:
                    break
                index += 1

    def get_kernel_dict(self) -> dict:
        acl_to_npu_dict = CANNFileParser(self._profiler_path).get_acl_to_npu_data()
        if not acl_to_npu_dict:
            return acl_to_npu_dict
        dequeue_data_list = FwkFileParser(self._profiler_path).get_dequeue_data()
        return self.combine_kernel_dict(acl_to_npu_dict, dequeue_data_list)

    def get_step_range(self, root_node: TorchOpNode, kernel_dict: dict):
        if not kernel_dict:
            print_error_msg("Failed to get acl to npu flow events.")
            return []
        step_node_list = []
        for level1_node in root_node.child_node_list:
            if level1_node.is_profiler_step():
                step_node_list.append(level1_node)
        if not step_node_list:
            return []
        if not FwkFileParser(self._profiler_path).has_task_queue_data():
            acl_start_time_list = sorted(list(kernel_dict.keys()))
            self._update_step_node_info(step_node_list, acl_start_time_list)
        step_range = []
        for step_node in step_node_list:
            step_id = step_node.event.name.split("#")[-1]
            if not step_node.corr_id_total:
                print_error_msg("Some step lost the correlation id information.")
                return []
            corr_id_list = sorted(step_node.corr_id_total)
            min_index, max_index = 0, len(corr_id_list) - 1
            min_kernel_list, max_kernel_list = [], []
            while min_index < len(corr_id_list):
                min_kernel_list = kernel_dict.get(corr_id_list[min_index], [])
                if min_kernel_list:
                    break
                min_index += 1
            while max_index >= 0:
                max_kernel_list = kernel_dict.get(corr_id_list[max_index], [])
                if max_kernel_list:
                    break
                max_index -= 1

            device_start_ts = min(
                [kernel.ts for kernel in min_kernel_list]) if min_kernel_list else step_node.start_time
            device_end_ts = max(
                [kernel.end_ns for kernel in max_kernel_list]) if max_kernel_list else Constant.INVALID_VALUE
            step_range.append(
                {Constant.STEP_ID: step_id, Constant.START_TS: device_start_ts,
                 Constant.END_TS: max(device_end_ts, step_node.end_time), Constant.COMM_OPS: {}})
        return step_range
