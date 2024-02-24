import os

from ..prof_common_func.constant import convert_ns2us_float
from .base_parser import BaseParser
from ..prof_bean.torch_op_node import TorchOpNode
from ..prof_common_func.constant import Constant, print_error_msg
from ..prof_common_func.constant import print_warn_msg
from ..prof_common_func.path_manager import ProfilerPathManager
from ..prof_common_func.tree_builder import TreeBuilder
from ..prof_parse.fwk_cann_relation_parser import FwkCANNRelationParser
from ..prof_parse.fwk_file_parser import FwkFileParser
from ....utils.path_manager import PathManager

__all__ = []


class StackViewParser(BaseParser):
    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._torch_op_node = []
        self._root_node = None
        self._kernel_dict = {}
        self._metric = param_dict.get("metric")

    def run(self, deps_data: dict):
        try:
            self._torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
            self.generate_view()
        except Exception:
            print_error_msg("Failed to export stack.")
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        self._init_data()
        if not self._torch_op_node:
            return
        output_path = os.path.realpath(self._output_path)
        parent_dir = os.path.dirname(self._output_path)
        PathManager.make_dir_safety(parent_dir)
        PathManager.check_directory_path_writeable(parent_dir)
        file_name, suffix = os.path.splitext(output_path)
        if suffix != ".log":
            print_warn_msg("Input file is not log file. Change to log file.")
            output_path = file_name + ".log"
        with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT, Constant.FILE_AUTHORITY), "w") as f:
            for torch_op_node in self._torch_op_node:
                call_stack = torch_op_node.call_stack
                if not call_stack:
                    continue
                if self._metric == Constant.METRIC_CPU_TIME:
                    total_dur = convert_ns2us_float(torch_op_node.host_self_dur)
                else:
                    total_dur = self._calculate_npu_dur(torch_op_node)
                if float(total_dur) <= 0:
                    continue
                # remove ‘\n’ for each stack frame
                call_stack_list = list(map(lambda x: x.strip(), call_stack.split(";")))
                call_stack_list = list(reversed(call_stack_list))
                call_stack_str = ";".join(call_stack_list)
                f.write(call_stack_str + " " + str(int(total_dur)) + "\n")

    def _calculate_npu_dur(self, torch_op_node: TorchOpNode):
        kernel_self = []
        for corr_id in torch_op_node.corr_id_self:
            kernel_self.extend(self._kernel_dict.get(corr_id, []))
        return sum([float(kernel.dur) for kernel in kernel_self])

    def _init_data(self):
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            self._torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(only_fwk=True)
        if not self._torch_op_node:
            return
        self._root_node = self._torch_op_node[0]
        self._torch_op_node = self._torch_op_node[1:]

        if self._metric == Constant.METRIC_NPU_TIME:
            self._kernel_dict = FwkCANNRelationParser(self._profiler_path).get_kernel_dict()
            if not FwkFileParser(self._profiler_path).has_task_queue_data():
                for acl_ts in self._kernel_dict.keys():
                    TreeBuilder.update_tree_node_info(acl_ts, self._root_node)
