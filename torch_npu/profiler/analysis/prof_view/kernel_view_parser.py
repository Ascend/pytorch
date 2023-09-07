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

from ..prof_common_func.file_manager import FileManager
from ..prof_bean.op_summary_bean import OpSummaryBean
from ..prof_common_func.global_var import GlobalVar
from ..prof_parse.cann_file_parser import CANNFileParser, CANNDataEnum
from ..prof_view.base_view_parser import BaseViewParser
from ..profiler_config import ProfilerConfig


class KernelViewParser(BaseViewParser):
    SHOW_HEADERS = ["Op Name", "OP Type", "Task Type", "Task Start Time", "Task Duration(us)", "Task Wait Time(us)",
                    "Block Dim"]
    KERNEL_BASE_HEADERS = ["Name", "Type", "Accelerator Core", "Start Time(us)", "Duration(us)", "Wait Time(us)",
                           "Block Dim"]
    KERNEL_VIEW = "kernel_details.csv"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self, output_path: str) -> None:
        op_summary_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.OP_SUMMARY)
        summary_data = []
        output_headers = self.KERNEL_BASE_HEADERS
        for file_path in op_summary_file_set:
            all_data = FileManager.read_csv_file(file_path, OpSummaryBean)
            if all_data:
                OpSummaryBean.headers = \
                    all_data[0].all_headers if ProfilerConfig().is_all_kernel_headers() else self.SHOW_HEADERS
                output_headers = self._project_map_for_headers(OpSummaryBean.headers)
            if not GlobalVar.step_range:
                summary_data.extend([data.row for data in all_data])
                continue
            for data in all_data:
                step_id = None
                for step_data in GlobalVar.step_range:
                    if step_data[1] <= data.ts <= step_data[2]:
                        step_id = step_data[0]
                        break
                summary_data.append([step_id] + data.row)

        headers = ["Step Id"] + output_headers if GlobalVar.step_range else output_headers
        FileManager.create_csv_file(output_path, summary_data, self.KERNEL_VIEW, headers)

    def _project_map_for_headers(self, input_headers: list):
        project_map_dict = {self.SHOW_HEADERS[i]: self.KERNEL_BASE_HEADERS[i] for i in range(len(self.SHOW_HEADERS))}
        output_headers = []
        for header in input_headers:
            if header in project_map_dict:
                output_headers.append(project_map_dict.get(header))
            else:
                output_headers.append(header)
        return output_headers
