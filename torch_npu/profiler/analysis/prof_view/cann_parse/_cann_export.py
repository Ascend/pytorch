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
import pwd
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone

from torch_npu.utils._error_code import ErrCode, prof_error

from ..._profiler_config import ProfilerConfig
from ...prof_common_func._cann_package_manager import CannPackageManager, find_msprof_py_path
from ...prof_common_func._constant import (
    Constant,
    print_error_msg,
    print_info_msg,
    print_warn_msg,
)
from ...prof_common_func._file_manager import FileManager
from ...prof_common_func._log import ProfilerLogger
from ...prof_common_func._path_manager import ProfilerPathManager
from .._base_parser import BaseParser
from ._cann_command import build_command_log_path, save_command_output

__all__ = []


class CANNExportParser(BaseParser):
    COMMAND_SUCCESS = 0
    error_msg = (
        "Export CANN Profiling data failed, please verify that the ascend-toolkit is installed and "
        "set_env.sh is sourced. or you can execute the command to confirm the CANN Profiling "
        "export result: msprof --export=on"
    )

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        self.msprof_path = shutil.which("msprof")

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "CANNExportParser")
        self.logger = ProfilerLogger.get_instance()
        self.logger.info("CANNExportParser start.")
        try:
            ProfilerConfig().load_info(self._profiler_path)
            if not os.path.isdir(self._cann_path):
                return Constant.SUCCESS, None
            self._check_msprof_environment()
            self._check_prof_data_size()
            start_time = datetime.now(tz=timezone.utc).astimezone()
            command_log_path = build_command_log_path(self._profiler_path, "export")

            if Constant.Db in self._export_type:
                analyze_cmd_list = [
                    self.msprof_path,
                    "--export=on",
                    "--type=db",
                    f"--output={self._cann_path}",
                ]
                completed_analysis = subprocess.run(
                    analyze_cmd_list, capture_output=True, shell=False,
                    text=True, encoding="utf-8", errors="replace",
                )
                save_command_output(command_log_path, analyze_cmd_list, completed_analysis)
                if completed_analysis.returncode != self.COMMAND_SUCCESS:
                    raise RuntimeError(
                        "Failed to export CANN DB Profiling data."
                        + prof_error(ErrCode.INTERNAL)
                    )

            if Constant.Text in self._export_type:
                # 避免老CANN包无type参数报错。新解析包支持 --type=all，一次调用内部并行导出
                # timeline 和 summary，交付件完成顺序不再固定，由 CANNTimelineParser 通过
                # finish.txt 判断 msprof timeline 是否可读。
                analyze_cmd_list = [
                    self.msprof_path,
                    "--export=on",
                ]
                if CannPackageManager.is_support_export_all():
                    analyze_cmd_list.append("--type=all")
                analyze_cmd_list.append(f"--output={self._cann_path}")
                completed_analysis = subprocess.run(
                    analyze_cmd_list, capture_output=True, shell=False,
                    text=True, encoding="utf-8", errors="replace",
                )
                save_command_output(command_log_path, analyze_cmd_list, completed_analysis)
                if completed_analysis.returncode != self.COMMAND_SUCCESS:
                    raise RuntimeError(
                        "Failed to export CANN TEXT Profiling data."
                        + prof_error(ErrCode.INTERNAL)
                    )

        except Exception as err:
            print_error_msg(f"Failed to export CANN Profiling data. Error msg: {err}")
            self.logger.exception("Failed to export CANN Profiling data")
            return Constant.FAIL, None
        end_time = datetime.now(tz=timezone.utc).astimezone()
        print_info_msg(
            f"CANN profiling data parsed in a total time of {end_time - start_time}"
        )
        self.logger.info("CANNExportParser finish.")
        return Constant.SUCCESS, None

    def _check_msprof_environment(self):
        self._check_msprof_profile_path_is_valid()
        self._check_msprof_path()

    def _check_msprof_profile_path_is_valid(self):
        self._check_profiler_path_parent_dir_invalid(
            ProfilerPathManager.get_all_subdir(self._cann_path)
        )

    def _check_profiler_path_parent_dir_invalid(self, paths: list):
        for path in paths:
            if not FileManager.check_file_owner(path):
                self.logger.warning(
                    "The owner of path '%s' is neither root nor the current user. "
                    "Please run: chown -R $(id -un) '%s'.",
                    self._cann_path, self._cann_path
                )
        return False

    def _check_msprof_path(self):
        error_message = ""
        if not self.msprof_path:
            error_message += (
                "Export CANN Profiling data failed! 'msprof' command not found!"
                "Please re-source the CANN package to fix this error. "
                + prof_error(ErrCode.NOT_FOUND)
                + "\n"
            )
            raise RuntimeError(error_message)
        if not ProfilerPathManager.check_path_permission(self.msprof_path):
            error_message += (
                f"The '{self.msprof_path}' path and current owner have inconsistent permissions."
                f"please execute 'chown -R {pwd.getpwuid(os.getuid()).pw_name} "
                f"{os.path.normpath(os.path.join(self.msprof_path, '../../..'))}'"
                + prof_error(ErrCode.PERMISSION)
                + "\n"
            )
        msprof_script_path = self._get_msprof_script_path()
        if not msprof_script_path:
            error_message += (
                f"Failed to find msprof.py path in {self.msprof_path}. Please check the CANN environment."
                + prof_error(ErrCode.NOT_FOUND)
                + "\n"
            )
            raise RuntimeError(error_message)
            # The path "os.path.join(msprof_script_path, '../../../../../../..')"  represents the layer of "ascend-toolkit",
            # such as "xxx/ascend-toolkit"
        if not ProfilerPathManager.check_path_permission(msprof_script_path):
            error_message += (
                f"The '{msprof_script_path}' path and current owner have inconsistent permissions."
                f"please execute 'chown -R {pwd.getpwuid(os.getuid()).pw_name} "
                f"{os.path.normpath(os.path.join(msprof_script_path, '../../../../../../..'))}'"
                + prof_error(ErrCode.PERMISSION)
                + "\n"
            )
        if error_message:
            self.logger.warning(error_message)

    def _get_msprof_script_path(self) -> str:
        return find_msprof_py_path(self.msprof_path)

    def _check_prof_data_size(self):
        if not self._cann_path:
            return
        device_paths = ProfilerPathManager.get_device_path(self._cann_path)
        prof_data_size = 0
        host_data_path = os.path.join(self._cann_path, "host", "data")
        for root, _, files in ProfilerPathManager.walk_with_depth(host_data_path):
            prof_data_size += sum(
                [os.path.getsize(os.path.join(root, name)) for name in files]
            )
        if not device_paths and prof_data_size < Constant.PROF_WARN_SIZE:
            return
        for device_path in device_paths:
            device_data_path = os.path.join(device_path, "data")
            for root, _, files in ProfilerPathManager.walk_with_depth(device_data_path):
                prof_data_size += sum(
                    [os.path.getsize(os.path.join(root, name)) for name in files]
                )
            if prof_data_size >= Constant.PROF_WARN_SIZE:
                print_warn_msg(
                    "The parsing time is expected to exceed 30 minutes, "
                    "and you can choose to stop the process and use offline parsing."
                )
                return


class CANNTimelineParser(BaseParser):
    # 解析侧导出时写出的完成标记文件：每完成一个交付件就写入其文件名。
    FINISH_FILE = "finish.txt"
    # finish.txt 中的整体结束标记行：所有 csv 解析结束后写入，表示不会再有新交付件。
    DONE_MARK = "done"
    MSPROF_JSON_PATTERN = r"^msprof.*\.json$"
    DB_FILE_PATTERN = r"^msprof_\d+\.db$"
    # 在 PROF 目录本身及其下一级目录中查找完成标记，避免绑定固定目录，后续位置调整也能找到。
    FINISH_FILE_SEARCH_DEPTH = 1
    # 查找完成标记的最小间隔：导出期间 PROF 目录下文件很多，避免每轮轮询都遍历目录。
    FINISH_FILE_SEARCH_INTERVAL = 1.0

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        self._cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        self._output_done_path = ""
        self._last_search_time = 0.0

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "CANNTimelineParser")
        self.logger = ProfilerLogger.get_instance()
        self.logger.info("CANNTimelineParser start.")
        if not os.path.isdir(self._cann_path):
            return Constant.SUCCESS, None
        ProfilerConfig().load_info(self._profiler_path)
        if Constant.Text not in self._export_type:
            return self._wait_for_db_file()
        if CannPackageManager.is_support_export_all():
            return self._wait_for_export_all_done()
        return self._wait_for_summary_file()

    def _wait_for_export_all_done(self):
        # --type=all 内部并行导出 timeline 和 summary，无法再用 csv 判断 timeline 是否就绪，
        # 改为等待 finish.txt 中出现 msprof 开头、.json 结尾的交付件：出现即说明 timeline
        # 已经落盘，下游解析可以提前开始，不必等 summary 一起结束。
        while True:
            output_done_lines = self._get_finish_lines()
            if output_done_lines is not None:
                if self._has_msprof_timeline(output_done_lines):
                    self.logger.info("CANNTimelineParser finish.")
                    return Constant.SUCCESS, None
                if self.DONE_MARK in output_done_lines:
                    self.logger.warning(
                        "No msprof timeline json file was exported, "
                        "the CANN timeline data is unavailable."
                    )
                    return Constant.SUCCESS, None
            try:
                time.sleep(Constant.SLEEP_TIME)
            except InterruptedError:
                return Constant.FAIL, None

    def _get_finish_lines(self):
        finish_path = self._find_finish_file()
        if not finish_path:
            return None
        try:
            content = FileManager.file_read_all(finish_path, "rt")
        except Exception:
            self.logger.warning("Failed to read %s.", self.FINISH_FILE)
            return None
        return [line.strip() for line in content.splitlines() if line.strip()]

    def _find_finish_file(self) -> str:
        if self._output_done_path:
            return self._output_done_path
        if not self._cann_path:
            return ""
        current_time = time.monotonic()
        if current_time - self._last_search_time < self.FINISH_FILE_SEARCH_INTERVAL:
            return ""
        self._last_search_time = current_time
        for root, _, files in ProfilerPathManager.walk_with_depth(
            self._cann_path, self.FINISH_FILE_SEARCH_DEPTH
        ):
            if self.FINISH_FILE in files:
                self._output_done_path = os.path.join(root, self.FINISH_FILE)
                return self._output_done_path
        return ""

    def _db_file_exists(self) -> bool:
        if not self._cann_path:
            return False
        for file in os.listdir(self._cann_path):
            if re.match(self.DB_FILE_PATTERN, file) and os.path.isfile(
                os.path.join(self._cann_path, file)
            ):
                return True
        return False

    @classmethod
    def _has_msprof_timeline(cls, output_done_lines: list) -> bool:
        return any(
            re.match(cls.MSPROF_JSON_PATTERN, file_name)
            for file_name in output_done_lines
        )

    def _wait_for_summary_file(self):
        # 老解析包按 timeline、summary 顺序导出，summary 的 csv 出现时 timeline json 已就绪。
        output_path = os.path.join(self._cann_path, "mindstudio_profiler_output")
        while True:
            if os.path.exists(output_path):
                for file_name in os.listdir(output_path):
                    if file_name.endswith(".csv"):
                        self.logger.info("CANNTimelineParser finish.")
                        return Constant.SUCCESS, None
            try:
                time.sleep(Constant.SLEEP_TIME)
            except InterruptedError:
                return Constant.FAIL, None

    def _wait_for_db_file(self):
        # db 导出路径不产出 msprof json，保持原有判定：db 文件出现即放行。
        # db 数据完整性由 CANNExportParser 等待 msprof 进程退出保证，不依赖本处信号。
        while True:
            if self._db_file_exists():
                self.logger.info("CANNTimelineParser finish.")
                return Constant.SUCCESS, None
            try:
                time.sleep(Constant.SLEEP_TIME)
            except InterruptedError:
                return Constant.FAIL, None
