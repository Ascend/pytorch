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

from ...prof_common_func._constant import Constant
from ...prof_common_func._file_manager import FileManager
from ...prof_common_func._log import ProfilerLogger
from ...prof_parse._fwk_file_parser import FwkFileParser
from .._base_parser import BaseParser

__all__ = []


class TracePreParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "TracePreParser")
        self.logger = ProfilerLogger.get_instance()
        try:
            fwk_trace_data = FwkFileParser(self._profiler_path).get_fwk_trace_data()
            trace_file_path = os.path.join(self._output_path, Constant.TRACE_VIEW_TEMP) if os.path.isdir(
                self._output_path) else self._output_path
            FileManager.create_prepare_trace_json_by_path(trace_file_path, fwk_trace_data)
        except Exception as e:
            self.logger.error("Failed to create prepare trace json, error: %s", str(e), exc_info=True)
            return Constant.FAIL, None
        return Constant.SUCCESS, None


class TreeBuildParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        ProfilerLogger.init(self._profiler_path, "TracePreParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        try:
            torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node()
        except Exception as e:
            self.logger.error("Failed to build torch op tree, error: %s", str(e), exc_info=True)
            return Constant.FAIL, []
        return Constant.SUCCESS, torch_op_node
