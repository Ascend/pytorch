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

from ...prof_common_func.constant import print_error_msg, Constant
from ...prof_parse.fwk_file_parser import FwkFileParser
from ...prof_view.base_parser import BaseParser


class TracePreParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        try:
            fwk_trace_data = FwkFileParser(self._profiler_path).get_fwk_trace_data()
        except Exception:
            print_error_msg("Failed to preprocess framework trace data.")
            return Constant.FAIL, []
        return Constant.SUCCESS, fwk_trace_data


class TreeBuildParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        try:
            torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node()
        except Exception:
            print_error_msg("Failed to build torch op tree.")
            return Constant.FAIL, []
        return Constant.SUCCESS, torch_op_node
