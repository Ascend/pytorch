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

from ..prof_common_func.global_var import GlobalVar
from ..prof_view.base_view_parser import BaseViewParser
from ..prof_common_func.constant import Constant
from ..prof_common_func.constant import print_warn_msg
from ....utils.path_manager import PathManager


class StackViewParser(BaseViewParser):
    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self, output_path: str, **kwargs) -> None:
        if not GlobalVar.torch_op_tree_node:
            return
        output_path = os.path.realpath(output_path)
        parent_dir = os.path.dirname(output_path)
        PathManager.make_dir_safety(parent_dir)
        PathManager.check_directory_path_writeable(parent_dir)
        file_name, suffix = os.path.splitext(output_path)
        if suffix != ".log":
            print_warn_msg("Input file is not log file. Change to log file.")
            output_path = file_name + ".log"
        metric = kwargs.get("metric")
        with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT, Constant.FILE_AUTHORITY), "w") as f:
            for torch_op_node in GlobalVar.torch_op_tree_node:
                call_stack = torch_op_node.call_stack
                if not call_stack:
                    continue
                if metric == Constant.METRIC_CPU_TIME:
                    total_dur = torch_op_node.host_self_dur
                else:
                    total_dur = torch_op_node.device_self_dur
                if float(total_dur) <= 0:
                    continue
                total_dur = round(float(total_dur))
                # remove ‘\n’ for each stack frame
                call_stack_list = list(map(lambda x: x.strip(), call_stack.split(";")))
                call_stack_list = list(reversed(call_stack_list))
                call_stack_str = ";".join(call_stack_list)
                f.write(call_stack_str + " " + str(total_dur) + "\n")
