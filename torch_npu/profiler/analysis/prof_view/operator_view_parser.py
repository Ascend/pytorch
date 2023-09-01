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
from ..prof_common_func.global_var import GlobalVar
from ..prof_view.base_view_parser import BaseViewParser


class OperatorViewParser(BaseViewParser):
    OPERATOR_HEADERS = ["Name", "Input Shapes", "Call Stack", "Host Self Duration(us)", "Host Total Duration(us)",
                        "Device Self Duration(us)", "Device Total Duration(us)", "Device Self Duration With AICore(us)",
                        "Device Total Duration With AICore(us)"]
    OPERATOR_VIEW = "operator_details.csv"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self, output_path: str) -> None:
        if not GlobalVar.torch_op_tree_node:
            return
        operator_list = [None] * len(GlobalVar.torch_op_tree_node)
        index = 0
        for torch_op_node in GlobalVar.torch_op_tree_node:
            if torch_op_node.is_profiler_step():
                continue
            operator_list[index] = [torch_op_node.event.name, torch_op_node.input_shape, torch_op_node.call_stack,
                                    torch_op_node.host_self_dur, torch_op_node.host_total_dur,
                                    torch_op_node.device_self_dur, torch_op_node.device_total_dur,
                                    torch_op_node.device_self_dur_with_ai_core,
                                    torch_op_node.device_total_dur_with_ai_core]
            index += 1
        del operator_list[index:]
        FileManager.create_csv_file(output_path, operator_list, self.OPERATOR_VIEW, self.OPERATOR_HEADERS)
