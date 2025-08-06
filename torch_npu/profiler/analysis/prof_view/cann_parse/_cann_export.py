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
from ...prof_common_func._file_manager import FileManager
from .._base_parser import BaseParser
from ..._profiler_config import ProfilerConfig
from ...prof_common_func._log import ProfilerLogger


__all__ = []


class CANNExportParser(BaseParser):
    COMMAND_SUCCESS = 0
    error_msg = f"Export CANN Profiling data failed, please verify that the ascend-toolkit is installed and " \
                f"set-env.sh is sourced. or you can execute the command to confirm the CANN Profiling " \
                f"export result: msprof --export=on"
    _MSPROF_PY_PATH = "tools/profiler/profiler_tool/analysis/msprof/msprof.py"

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
            self._check_msprof_environment()
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

    def _check_msprof_environment(self):
        self._check_msprof_profile_path_is_valid()
        self._check_msprof_cmd_path_exist()
        self._check_msprof_cmd_path_permission()
        self._check_msprof_py_path_permission()

    def _check_msprof_profile_path_is_valid(self):
        self._check_profiler_path_parent_dir_invalid(ProfilerPathManager.get_all_subdir(self._cann_path))

    def _check_profiler_path_parent_dir_invalid(self, paths: list):
        for path in paths:
            if not FileManager.check_file_owner(path):
                raise RuntimeError(f"Path '{self._cann_path}' owner is neither root nor the current user. "
                                   f"Please execute 'chown -R $(id -un) '{self._cann_path}' '.")
            if ProfilerPathManager.path_is_other_writable(path):
                raise RuntimeError(f"Path '{self._cann_path}' permission allow others users to write. "
                                   f"Please execute 'chmod -R 755 '{self._cann_path}' '.")
        return False

    def _check_msprof_cmd_path_exist(self):
        if not self.msprof_path:
            raise RuntimeError("Export CANN Profiling data failed! 'msprof' command not found!"
                               + prof_error(ErrCode.NOT_FOUND))

    def _check_msprof_cmd_path_permission(self):
        ProfilerPathManager.check_path_permission(self.msprof_path)

    def _check_msprof_py_path_permission(self):
        msprof_script_path = self._get_msprof_script_path(self._MSPROF_PY_PATH)
        if not msprof_script_path:
            raise FileNotFoundError(
                "Failed to find msprof.py path. Please check the CANN environment."
            )
        ProfilerPathManager.check_path_permission(msprof_script_path)

    def _get_msprof_script_path(self, script_path: str) -> str:
        msprof_path = os.path.realpath(self.msprof_path.strip())
        pre_path = msprof_path.split("tools")[0]
        full_script_path = os.path.join(pre_path, script_path)
        return full_script_path if os.path.exists(full_script_path) else ""

    def _check_prof_data_size(self):
        if not self._cann_path:
            return
        device_paths = ProfilerPathManager.get_device_path(self._cann_path)
        prof_data_size = 0
        host_data_path = os.path.join(self._cann_path, "host", "data")
        for root, _, files in os.walk(host_data_path):
            prof_data_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        if not device_paths and prof_data_size < Constant.PROF_WARN_SIZE:
            return
        for device_path in device_paths:
            device_data_path = os.path.join(device_path, "data")
            for root, _, files in os.walk(device_data_path):
                prof_data_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
            if prof_data_size >= Constant.PROF_WARN_SIZE:
                print_warn_msg("The parsing time is expected to exceed 30 minutes, "
                               "and you can choose to stop the process and use offline parsing.")
                return


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
