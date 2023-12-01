import os

from .base_parser import BaseParser
from ..prof_common_func.constant import Constant, print_error_msg
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.path_manager import ProfilerPathManager
from ..prof_common_func.trace_event_manager import TraceEventManager
from ..prof_common_func.tree_builder import TreeBuilder
from ..prof_parse.fwk_cann_relation_parser import FwkCANNRelationParser
from ..profiler_config import ProfilerConfig
from ..prof_parse.cann_file_parser import CANNFileParser
from ..prof_parse.fwk_file_parser import FwkFileParser


class TraceViewParser(BaseParser):
    TRACE_VIEW = "trace_view.json"

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._trace_file_path = os.path.join(self._output_path, self.TRACE_VIEW) if os.path.isdir(
            self._output_path) else self._output_path
        self._temp_trace_file_path = os.path.join(self._output_path, Constant.TRACE_VIEW_TEMP) if os.path.isdir(
            self._output_path) else self._output_path
        self._trace_data = []
        self._torch_op_node = []
        self._root_node = None

    @staticmethod
    def _prune_trace_by_level(json_data: list) -> list:
        prune_config = ProfilerConfig().get_prune_config()
        if not prune_config or not json_data:
            return json_data
        result = []
        for data in json_data:
            prune_flag = False
            for prune_key in prune_config:
                if data.get("name", "").startswith(prune_key) or data.get("args", {}).get("name", "") == prune_key:
                    prune_flag = True
                    continue
            if not prune_flag:
                result.append(data)
        return result

    def run(self, deps_data: dict):
        try:
            ProfilerConfig().load_info(self._profiler_path)
            torch_op_node = deps_data.get(Constant.TREE_BUILD_PARSER, [])
            if torch_op_node:
                self._root_node = torch_op_node[0]
                self._torch_op_node = torch_op_node[1:]
            self.generate_view()
        except Exception:
            print_error_msg("Failed to generate trace_view.json.")
            return Constant.FAIL, None
        return Constant.SUCCESS, None

    def generate_view(self) -> None:
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            self._trace_data = FwkFileParser(self._profiler_path).get_fwk_trace_data()
        else:
            msprof_timeline_data = CANNFileParser(self._profiler_path).get_timeline_all_data()
            self._trace_data.extend(
                self._prune_trace_by_level(msprof_timeline_data))
            if self._torch_op_node:
                self._trace_data.extend(self._get_flow_event(msprof_timeline_data))
        if os.path.exists(self._temp_trace_file_path):
            FileManager.append_trace_json_by_path(self._temp_trace_file_path, self._trace_data, self._trace_file_path)
        else:
            FileManager.create_json_file_by_path(self._trace_file_path, self._trace_data)

    def _get_flow_event(self, msprof_timeline_data: list) -> list:
        flow_event_list = []
        acl_to_npu_dict = CANNFileParser.combine_acl_to_npu(msprof_timeline_data)
        if not FwkFileParser(self._profiler_path).has_task_queue_data():
            for acl_ts in acl_to_npu_dict.keys():
                matched_torch_op = TreeBuilder.match_self_torch_op(acl_ts, self._root_node)
                if not matched_torch_op:
                    continue
                kernel_list = acl_to_npu_dict.get(acl_ts, [])
                for kernel in kernel_list:
                    flow_event_list.extend(
                        TraceEventManager.create_torch_to_npu_flow(matched_torch_op.event, kernel))
            return flow_event_list
        dequeue_data_list = FwkFileParser(self._profiler_path).get_dequeue_data()
        kernel_dict = FwkCANNRelationParser.combine_kernel_dict(acl_to_npu_dict, dequeue_data_list)
        for torch_op_node in self._torch_op_node:
            for corr_id in torch_op_node.corr_id_self:
                kernel_list = kernel_dict.get(corr_id, [])
                for kernel in kernel_list:
                    flow_event_list.extend(TraceEventManager.create_torch_to_npu_flow(torch_op_node.event, kernel))
        return flow_event_list
