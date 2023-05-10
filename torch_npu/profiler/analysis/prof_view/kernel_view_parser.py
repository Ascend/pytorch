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
import re

from ..prof_common_func.constant import print_warn
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.file_tag import FileTag
from ..prof_bean.op_summary_bean import OpSummaryBean
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum
from ..prof_parse.fwk_file_parser import FwkFileParser
from ..prof_view.base_view_parser import BaseViewParser


class KernelViewParser(BaseViewParser):
    KERNEL_VIEW = "kernel_details.csv"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self: any, output_path: str = None) -> None:
        step_id_dict = self._get_step_id_dict()
        op_summary_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.OP_SUMMARY)
        summary_data = []
        has_headers = False
        step_id = []
        for file_path in op_summary_file_set:
            if len(os.path.basename(file_path).split(".")[0].split("_")) == 5:
                iter_id = int(os.path.basename(file_path).split(".")[0].split("_")[-1])
                step_id = [step_id_dict.get(iter_id, None)] if step_id_dict.get(iter_id) is not None else []
            all_data = FileManager.read_csv_file(file_path, OpSummaryBean)
            for data in all_data:
                if not has_headers:
                    summary_data.append(data.headers(step_id))
                    has_headers = True
                summary_data.append(step_id + data.row)
        FileManager.create_csv_file(self._profiler_path, summary_data, self.KERNEL_VIEW)

    def _get_step_id_dict(self: any) -> dict:
        step_id_dict = {}
        step_id_list = []
        torch_op_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.TORCH_OP)
        for torch_op in torch_op_data:
            if torch_op.name.find("ProfilerStep#") != -1:
                try:
                    step_id_list.append(int(torch_op.name.split("#")[-1]))
                except ValueError:
                    print_warn("Invalid step id")
        step_id_list.sort()
        for key, value in enumerate(step_id_list):
            # cann iteration id start from 1
            step_id_dict[key + 1] = value
        return step_id_dict
