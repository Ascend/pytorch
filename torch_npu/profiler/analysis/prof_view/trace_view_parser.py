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
        prune_process = ProfilerConfig().get_prune_process()
        if not prune_config or not json_data:
            return json_data
        result = []
        prune_pid = []
        for data in json_data:
            if data.get("name", "") == "process_name" and data.get("args", {}).get("name", "") in prune_process:
                prune_pid.append(data.get("pid", ""))
        for data in json_data:
            prune_flag = False
            if data.get("pid") in prune_pid:
                continue
            for prune_key in prune_config:
                if data.get("name", "").startswith(prune_key) or data.get("args", {}).get("name", "") == prune_key:
                    prune_flag = True
                    break
            if not prune_flag:
                result.append(data)
        return result

    def generate_view(self, output_path: str, **kwargs) -> None:
        trace_data = self._prune_trace_by_level(CANNFileParser(self._profiler_path).get_timeline_all_data())
        self._add_fwk_trace_data(trace_data)
        if os.path.isdir(output_path):
            FileManager.create_json_file(output_path, trace_data, self.TRACE_VIEW)
            TraceStepTimeParser.create_step_file(output_path, trace_data, self.STEP_TRACE)
        else:
            FileManager.create_json_file_by_path(output_path, trace_data)

    def deal_sequence_one_date(self, seqnum, torch_op, flow_node, fwd_dct, mode):
        if flow_node and flow_node.get(mode) and flow_node.get(mode).get('ts') > torch_op.event.ts:
            return
        else:
            start_node = {seqnum:{mode: {'pid': torch_op.event.pid, 'tid': torch_op.event.tid, 'ts': torch_op.event.ts}}}
            if flow_node:
                fwd_dct.get(seqnum).update(start_node.get(seqnum))
            else:
                fwd_dct.update(start_node)

    def get_sequence_trace_data(self):
        if not GlobalVar.torch_op_tree_node:
            return []
        fwd_dct = {}
        for torch_op in GlobalVar.torch_op_tree_node:
            seqnum = torch_op.event.args.get("Sequence number", -1)
            if seqnum > 0:
                flow_node = fwd_dct.get(seqnum)
                if torch_op.event.args.get("Fwd thread id") == 0:
                    self.deal_sequence_one_date(seqnum, torch_op, flow_node, fwd_dct, 'start')
                else:
                    self.deal_sequence_one_date(seqnum, torch_op, flow_node, fwd_dct, 'end')
        return TraceEventManager.create_fwd_flow(fwd_dct)

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

        fwd_list = self.get_sequence_trace_data()
        json_data.extend(fwk_x_event_list + fwk_other_event_list + fwd_list)
