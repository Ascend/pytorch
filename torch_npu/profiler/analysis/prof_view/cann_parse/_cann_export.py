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
import shutil
import subprocess
import time
from datetime import datetime

from torch_npu.utils._error_code import ErrCode, prof_error
from ...prof_common_func._constant import Constant, print_warn_msg, print_error_msg, print_info_msg
from ...prof_common_func._path_manager import ProfilerPathManager
from .._base_parser import BaseParser
from ..._profiler_config import ProfilerConfig
from ...prof_common_func._log import ProfilerLogger


__all__ = []


class CANNExportParser(BaseParser):
    COMMAND_SUCCESS = 0
    error_msg = f"Export CANN Profiling data failed, please verify that the ascend-toolkit is installed and " \
                f"set-env.sh is sourced. or you can execute the command to confirm the CANN Profiling " \
                f"export result: msprof --export=on"

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        self.msprof_path = shutil.which("msprof")

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "CANNExportParser")
        self.logger = ProfilerLogger.get_instance()
        try:
            ProfilerConfig().load_info(self._profiler_path)
            if not os.path.isdir(self._cann_path):
                return Constant.SUCCESS, None
            if not self.msprof_path:
                err_msg = "Export CANN Profiling data failed! msprof command not found!" + prof_error(ErrCode.NOT_FOUND)
                print_error_msg(err_msg)
                raise RuntimeError(err_msg)
            self._check_prof_data_size()
            start_time = datetime.utcnow()

            if Constant.Db in self._export_type:
                analyze_cmd_list = [self.msprof_path, "--export=on", "--type=db", f"--output={self._cann_path}"]
                completed_analysis = subprocess.run(analyze_cmd_list, capture_output=True, shell=False)
                if completed_analysis.returncode != self.COMMAND_SUCCESS:
                    raise RuntimeError("Failed to export CANN DB Profiling data." + prof_error(ErrCode.INTERNAL))

            if Constant.Text in self._export_type:
                # 避免老CANN包无type参数报错
                analyze_cmd_list = [self.msprof_path, "--export=on", f"--output={self._cann_path}"]
                completed_analysis = subprocess.run(analyze_cmd_list, capture_output=True, shell=False)
                if completed_analysis.returncode != self.COMMAND_SUCCESS:
                    raise RuntimeError("Failed to export CANN TEXT Profiling data." + prof_error(ErrCode.INTERNAL))

        except Exception as err:
            print_error_msg(f"Failed to export CANN Profiling data. Error msg: {err}")
            self.logger.error("Failed to export CANN Profiling data, error: %s", str(err), exc_info=True)
            return Constant.FAIL, None
        end_time = datetime.utcnow()
        print_info_msg(f"CANN profiling data parsed in a total time of {end_time - start_time}")
        return Constant.SUCCESS, None

    def _check_prof_data_size(self):
        if not self._cann_path:
            return
        device_data_path = os.path.join(ProfilerPathManager.get_device_path(self._cann_path), "data")
        host_data_path = os.path.join(self._cann_path, "host", "data")
        prof_data_size = 0
        for root, dirs, files in os.walk(device_data_path):
            prof_data_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        for root, dirs, files in os.walk(host_data_path):
            prof_data_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        if prof_data_size >= Constant.PROF_WARN_SIZE:
            print_warn_msg("The parsing time is expected to exceed 30 minutes, "
                           "and you can choose to stop the process and use offline parsing.")


class CANNTimelineParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)

    def run(self, deps_data: dict):
        if not os.path.isdir(self._cann_path):
            return Constant.SUCCESS, None
        ProfilerConfig().load_info(self._profiler_path)
        if Constant.Text in self._export_type:
            output_path = os.path.join(self._cann_path, "mindstudio_profiler_output")
            while True:
                if os.path.exists(output_path):
                    for file_name in os.listdir(output_path):
                        if file_name.endswith('.csv'):
                            return Constant.SUCCESS, None
                try:
                    time.sleep(Constant.SLEEP_TIME)
                except InterruptedError:
                    return Constant.FAIL, None
        else:
            patten = r'^msprof_\d+\.db$'
            while True:
                for file in os.listdir(self._cann_path):
                    if re.match(patten, file) and os.path.isfile(os.path.join(self._cann_path, file)):
                        return Constant.SUCCESS, None
                try:
                    time.sleep(Constant.SLEEP_TIME)
                except InterruptedError:
                    return Constant.FAIL, None
