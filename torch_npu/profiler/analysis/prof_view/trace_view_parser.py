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
from ..prof_common_func.file_tag import FileTag
from ..prof_parse.cann_file_parser import CANNFileParser
from ..prof_parse.fwk_cann_relation_parser import FwkCANNRelationParser
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_view.base_view_parser import BaseViewParser


class TraceViewParser(BaseViewParser):
    TRACE_VIEW = "trace_view.json"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self, output_path: str = None) -> None:
        timeline_data = CANNFileParser(self._profiler_path).get_timeline_all_data()
        timeline_data = self._filter_iteration_id(timeline_data)
        self._add_torch_to_acl_and_npu_line(timeline_data)
        self._add_fwk_timeline(timeline_data)
        if output_path:
            FileManager.create_json_file_by_path(output_path, timeline_data)
            return
        FileManager.create_json_file(self._profiler_path, timeline_data, self.TRACE_VIEW)

    def _filter_iteration_id(self, json_data: list) -> list:
        result_json = []
        for data in json_data:
            if data.get("name", "").find("Iteration") == -1:
                result_json.append(data)
        return result_json

    def _add_fwk_timeline(self, json_data: list) -> None:
        torch_op_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.TORCH_OP)

        if not torch_op_data:
            return
        enqueue_data_list, dequeue_data_list = FwkFileParser(self._profiler_path).get_task_queue_data()
        event_data_list = torch_op_data + enqueue_data_list + dequeue_data_list
        pid = None
        tid_set = set()
        torch_op_trace_list = []
        for event in event_data_list:
            pid = event.pid
            tid_set.add(event.tid)
            torch_op_trace_list.append({
                "ph": "X", "name": event.name, "pid": f"0_{event.pid}", "tid": event.tid, "ts": event.ts,
                "dur": event.dur, "args": event.args
            })
        json_data.extend(torch_op_trace_list)
        json_data.extend(
            [{"ph": "M", "name": Constant.PROCESS_NAME, "pid": f"0_{pid}", "tid": 0, "args": {"name": "Python"}},
             {"ph": "M", "name": Constant.PROCESS_LABEL, "pid": f"0_{pid}", "tid": 0, "args": {"labels": "CPU"}},
             {"ph": "M", "name": Constant.PROCESS_SORT, "pid": f"0_{pid}", "tid": 0, "args": {"sort_index": 0}}])
        for tid in tid_set:
            if dequeue_data_list and dequeue_data_list[0].tid == tid:
                sort_index = max(tid_set) + 1
            else:
                sort_index = tid
            json_data.extend([{"ph": "M", "name": Constant.THREAD_NAME, "pid": f"0_{pid}", "tid": tid,
                               "args": {"name": f"Thread {tid}"}},
                              {"ph": "M", "name": Constant.THREAD_SORT, "pid": f"0_{pid}", "tid": tid,
                               "args": {"sort_index": sort_index}}])

    def _add_torch_to_acl_and_npu_line(self, json_data: list) -> None:
        relation_data_list = FwkCANNRelationParser(self._profiler_path).get_relation_data()
        if not relation_data_list:
            return
        end_point_list = []
        for data in json_data:
            if data.get("name", "") in Constant.ACL_OP_EXE_NAME:
                end_point_list.append(
                    {"ph": "f", "bp": "e", "name": "torch_to_acl", "id": data.get("ts"), "pid": data.get("pid"),
                     "tid": data.get("tid"), "ts": data.get("ts"), "cat": "async_acl_npu"})
            elif data.get("name") == "acl_to_npu" and data.get("ph") == "f":
                end_point_list.append({"ph": "f", "bp": "e", "name": "torch_to_npu",
                                       "id": data.get("id", "").replace("-", "_"), "pid": data.get("pid"),
                                       "tid": data.get("tid"), "ts": data.get("ts"), "cat": "async_npu"})
        json_data.extend(end_point_list)

        for relation_data in relation_data_list:
            # torch to acl line start_point
            json_data.append({"ph": "s", "bp": "e", "name": "torch_to_acl", "id": relation_data.acl_start_time,
                              "pid": f"0_{relation_data.torch_op_pid}", "tid": relation_data.torch_op_tid,
                              "ts": relation_data.torch_op_start_time, "cat": "async_acl_npu"})
            # torch to npu line start_point
            for kernel_id in relation_data.npu_kernel_list:
                json_data.append({"ph": "s", "bp": "e", "name": "torch_to_npu", "id": kernel_id,
                                  "pid": f"0_{relation_data.torch_op_pid}", "tid": relation_data.torch_op_tid,
                                  "ts": relation_data.torch_op_start_time, "cat": "async_npu"})
