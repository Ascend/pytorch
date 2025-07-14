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
import shutil
import subprocess

from torch_npu.utils._error_code import ErrCode, prof_error
from ...prof_common_func._constant import print_warn_msg, Constant, print_error_msg
from ...prof_common_func._path_manager import ProfilerPathManager
from .._base_parser import BaseParser
from ...prof_common_func._log import ProfilerLogger
from ..._profiler_config import ProfilerConfig

__all__ = []


class CANNAnalyzeParser(BaseParser):
    COMMAND_SUCCESS = 0

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        self.msprof_path = shutil.which("msprof")

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "CANNAnalyzeParser")
        self.logger = ProfilerLogger.get_instance()
        try:
            ProfilerConfig().load_info(self._profiler_path)
            if not os.path.isdir(self._cann_path):
                return Constant.SUCCESS, None
            if not self.msprof_path:
                err_msg = "Export CANN Profiling data faile! msprof command not found!" + prof_error(ErrCode.NOT_FOUND)
                print_error_msg(err_msg)
                raise RuntimeError(err_msg)

            if Constant.Db in self._export_type:
                analyze_cmd_list = [self.msprof_path, "--analyze=on", "--type=db", f"--output={self._cann_path}"]
                completed_analysis = subprocess.run(analyze_cmd_list, capture_output=True, shell=False)
                if completed_analysis.returncode != self.COMMAND_SUCCESS:
                    print_warn_msg("Failed to analyze CANN DB Profiling data.")

            if Constant.Text in self._export_type:
                analyze_cmd_list = [self.msprof_path, "--analyze=on", f"--output={self._cann_path}"]
                completed_analysis = subprocess.run(analyze_cmd_list, capture_output=True, shell=False)
                if completed_analysis.returncode != self.COMMAND_SUCCESS:
                    print_warn_msg("Failed to analyze CANN TEXT Profiling data.")

        except Exception as e:
            print_error_msg("Failed to analyze CANN Profiling data.")
            self.logger.error("Failed to analyze CANN Profiling data, error: %s", str(e), exc_info=True)
            return Constant.FAIL, None
        return Constant.SUCCESS, None
