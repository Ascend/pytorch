import os
from ..prof_common_func.constant import Constant
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.global_var import GlobalVar
from ..prof_common_func.trace_event_manager import TraceEventManager
from ..profiler_config import ProfilerConfig
from ..prof_parse.cann_file_parser import CANNFileParser
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_view.base_view_parser import BaseViewParser
from ..prof_view.trace_step_time import TraceStepTimeParser


class TraceViewParser(BaseViewParser):
    TRACE_VIEW = "trace_view.json"
    STEP_TRACE = "step_trace_time.csv"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

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

    def generate_view(self, output_path: str, **kwargs) -> None:
        trace_data = self._prune_trace_by_level(CANNFileParser(self._profiler_path).get_timeline_all_data())
        self._add_fwk_trace_data(trace_data)
        GlobalVar.torch_op_tree_node = []
        if os.path.isdir(output_path):
            FileManager.create_json_file(output_path, trace_data, self.TRACE_VIEW)
            TraceStepTimeParser.create_step_file(output_path, trace_data, self.STEP_TRACE)
        else:
            FileManager.create_json_file_by_path(output_path, trace_data)

    def _add_fwk_trace_data(self, json_data: list):
        if not GlobalVar.torch_op_tree_node:
            return
        pid = GlobalVar.torch_op_tree_node[0].event.pid
        tid_dict = {}
        enqueue_data_list, dequeue_data_list = FwkFileParser(self._profiler_path).get_task_queue_data()
        fwk_x_event_list = [None] * (
                len(GlobalVar.torch_op_tree_node) + len(enqueue_data_list) * 2 + len(dequeue_data_list) * 2)
        fwk_other_event_list = []
        index = 0
        for torch_op_node in GlobalVar.torch_op_tree_node:
            tid_dict[torch_op_node.event.tid] = False
            fwk_x_event_list[index] = TraceEventManager.create_x_event(torch_op_node.event, "cpu_op")
            index += 1
            if torch_op_node.kernel_list:
                for kernel in torch_op_node.kernel_list:
                    fwk_other_event_list.extend(TraceEventManager.create_torch_to_npu_flow(torch_op_node.event, kernel))

        for enqueue_data in enqueue_data_list:
            tid_dict[enqueue_data.tid] = False
            fwk_x_event_list[index] = TraceEventManager.create_x_event(enqueue_data, "enqueue")
            index += 1
            fwk_x_event_list[index] = TraceEventManager.create_task_queue_flow(Constant.FLOW_START_PH, enqueue_data)
            index += 1
        for dequeue_data in dequeue_data_list:
            tid_dict[dequeue_data.tid] = True
            fwk_x_event_list[index] = TraceEventManager.create_x_event(dequeue_data, "dequeue")
            index += 1
            fwk_x_event_list[index] = TraceEventManager.create_task_queue_flow(Constant.FLOW_END_PH, dequeue_data)
            index += 1
        fwk_other_event_list.extend(TraceEventManager.create_m_event(pid, tid_dict))

        json_data.extend(fwk_x_event_list + fwk_other_event_list)
