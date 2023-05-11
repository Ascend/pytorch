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
from warnings import warn

from ..prof_bean.event_bean import EventBean
from ..prof_common_func.file_tag import FileTag
from ..prof_bean.torch_acl_bean import TorchAclBean
from ..prof_parse.cann_file_parser import CANNFileParser
from ..prof_parse.fwk_file_parser import FwkFileParser


class FwkCANNRelationParser:
    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    @classmethod
    def _match_event(cls, sorted_event_list: list, sorted_mark_data_list: list) -> dict:
        result_dict = {}
        event_index, unmatched_num = 0, 0
        for mark_data in sorted_mark_data_list:
            matched_event = None
            while event_index < len(sorted_event_list):
                if sorted_event_list[event_index].ts > mark_data.ts:
                    break
                matched_event = sorted_event_list[event_index]
                event_index += 1
            if matched_event and matched_event.ts + matched_event.dur >= mark_data.ts:
                result_dict[mark_data.corr_id] = matched_event
            else:
                unmatched_num += 1
        if unmatched_num:
            warn(f"The number of unmatched events is: {unmatched_num}")
        return result_dict

    def get_relation_data(self) -> list:
        relation_data = []
        acl_to_npu_dict = CANNFileParser(self._profiler_path).get_acl_and_npu_data()
        torch_op_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_data or not acl_to_npu_dict:
            return relation_data

        acl_data_list = []
        for acl_start_time in acl_to_npu_dict.keys():
            acl_data_list.append(EventBean({"ts": acl_start_time, "corr_id": acl_start_time}))
        acl_data_list.sort(key=lambda x: x.ts)
        torch_op_data.sort(key=lambda x: x.ts)

        acl_to_torch_dict = {}
        enqueue_data_list, dequeue_data_list = FwkFileParser(self._profiler_path).get_task_queue_data()
        if enqueue_data_list and dequeue_data_list:
            acl_to_dequeue_dict = self._match_event(dequeue_data_list, acl_data_list)
            enqueue_to_torch_dict = self._match_event(torch_op_data, enqueue_data_list)
            for acl_start_time, dequeue_data in acl_to_dequeue_dict.items():
                acl_to_torch_dict[acl_start_time] = enqueue_to_torch_dict.get(dequeue_data.corr_id)
        else:
            acl_to_torch_dict = self._match_event(torch_op_data, acl_data_list)

        for acl_start_time, kernel_ids in acl_to_npu_dict.items():
            matched_torch_op = acl_to_torch_dict.get(acl_start_time)
            if matched_torch_op:
                relation_data.append(TorchAclBean(
                    {"torch_op_pid": matched_torch_op.pid, "torch_op_tid": matched_torch_op.tid,
                     "torch_op_start_time": matched_torch_op.ts, "op_name": matched_torch_op.name,
                     "acl_start_time": acl_start_time, "npu_kernel_list": kernel_ids}))
        return relation_data
