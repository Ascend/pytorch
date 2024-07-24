from ._base_parser import BaseParser
from ..prof_common_func._constant import Constant, print_error_msg
from ..prof_common_func._file_manager import FileManager

from ..prof_common_func._constant import convert_ns2us_float
from ..prof_common_func._path_manager import ProfilerPathManager
from ..prof_common_func._tree_builder import TreeBuilder
from ..prof_parse._fwk_file_parser import FwkFileParser

__all__ = []


class OperatorViewParser(BaseParser):
    OPERATOR_HEADERS = ["Name", "Input Shapes", "Call Stack", "Host Self Duration(us)", "Host Total Duration(us)",
                        "Device Self Duration(us)", "Device Total Duration(us)", "Device Self Duration With AICore(us)",
                        "Device Total Duration With AICore(us)"]
    OPERATOR_VIEW = "operator_details.csv"

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._torch_op_node = []
        self._root_node = None
        self._kernel_dict = {}

    def run(self, deps_data: dict):
        try:
            self._torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
            self._kernel_dict = deps_data.get(Constant.RELATION_PARSER, {})
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate operator_details.csv.")
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        self._init_torch_op()
        if not self._torch_op_node:
            return
        operator_list = [None] * len(self._torch_op_node)
        self._update_tree_for_no_task_queue()
        index = 0
        for torch_op_node in self._torch_op_node:
            if torch_op_node.is_profiler_step():
                continue
            kernel_self, kernel_total = [], []
            for corr_id in torch_op_node.corr_id_self:
                kernel_self.extend(self._kernel_dict.get(corr_id, []))
            for corr_id in torch_op_node.corr_id_total:
                kernel_total.extend(self._kernel_dict.get(corr_id, []))

            device_self_dur = sum([float(kernel.dur) for kernel in kernel_self])
            device_total_dur = sum([float(kernel.dur) for kernel in kernel_total])
            device_self_dur_with_ai_core = sum([float(kernel.dur) if kernel.is_ai_core else 0 for kernel in kernel_self])
            device_total_dur_with_ai_core = sum([float(kernel.dur) if kernel.is_ai_core else 0 for kernel in kernel_total])
            operator_list[index] = [torch_op_node.event.name, torch_op_node.input_shape, torch_op_node.call_stack,
                                    convert_ns2us_float(torch_op_node.host_self_dur),
                                    convert_ns2us_float(torch_op_node.host_total_dur), device_self_dur,
                                    device_total_dur, device_self_dur_with_ai_core, device_total_dur_with_ai_core]
            index += 1
        del operator_list[index:]
        FileManager.create_csv_file(self._output_path, operator_list, self.OPERATOR_VIEW, self.OPERATOR_HEADERS)

    def _update_tree_for_no_task_queue(self):
        if not FwkFileParser(self._profiler_path).has_task_queue_data():
            for acl_ts in self._kernel_dict.keys():
                TreeBuilder.update_tree_node_info(acl_ts, self._root_node)

    def _init_torch_op(self):
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            self._torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(only_fwk=True)
        if self._torch_op_node:
            self._root_node = self._torch_op_node[0]
            self._torch_op_node = self._torch_op_node[1:]
