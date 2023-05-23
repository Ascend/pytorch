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

from ..prof_common_func.constant import Constant
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.global_var import GlobalVar
from ..prof_common_func.trace_event_manager import TraceEventManager
from ..prof_parse.cann_file_parser import CANNFileParser
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_view.base_view_parser import BaseViewParser


class TraceViewParser(BaseViewParser):
    TRACE_VIEW = "trace_view.json"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self, output_path: str = None) -> None:
        trace_data = CANNFileParser(self._profiler_path).get_timeline_all_data()
        self._add_fwk_trace_data(trace_data)
        GlobalVar.torch_op_tree_node = []
        if output_path:
            FileManager.create_json_file_by_path(output_path, trace_data)
            return
        FileManager.create_json_file(self._profiler_path, trace_data, self.TRACE_VIEW)

    def _add_fwk_trace_data(self, json_data: list):
        if not GlobalVar.torch_op_tree_node:
            return
        acl_to_npu_dict = CANNFileParser(self._profiler_path).get_acl_to_npu_data()
        pid = GlobalVar.torch_op_tree_node[0].event.pid
        tid_dict = {}
        fwk_json_data = []

        for torch_op_node in GlobalVar.torch_op_tree_node:
            tid_dict[torch_op_node.event.tid] = False
            fwk_json_data.append(TraceEventManager.create_x_event(torch_op_node.event, "cpu_op"))
            if torch_op_node.acl_ts is not None:
                fwk_json_data.append(
                    TraceEventManager.create_torch_to_acl_flow_start(torch_op_node.acl_ts, torch_op_node.event))
                kernel_list = acl_to_npu_dict.get(torch_op_node.acl_ts, [])
                for kernel in kernel_list:
                    fwk_json_data.extend(TraceEventManager.create_torch_to_npu_flow(torch_op_node.event, kernel))

        enqueue_data_list, dequeue_data_list = FwkFileParser(self._profiler_path).get_task_queue_data()
        for queue_data in enqueue_data_list + dequeue_data_list:
            tid_dict[queue_data.tid] = queue_data.is_dequeue
            fwk_json_data.append(TraceEventManager.create_x_event(queue_data, "task_queue"))
        fwk_json_data.extend(TraceEventManager.create_m_event(pid, tid_dict))

        for data in json_data:
            if data.get("name", "").lower() in Constant.ACL_OP_EXE_NAME:
                fwk_json_data.append(
                    {"ph": "f", "bp": "e", "name": "torch_to_acl", "id": data.get("ts"), "pid": data.get("pid"),
                     "tid": data.get("tid"), "ts": data.get("ts"), "cat": "async_acl_npu"})

        json_data.extend(fwk_json_data)
