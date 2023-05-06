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

from ..prof_common_func.constant import Constant


class PathManager:
    @classmethod
    def get_fwk_path(cls, profiler_path: str) -> str:
        fwk_path = os.path.join(profiler_path, Constant.FRAMEWORK_DIR)
        if os.path.isdir(fwk_path):
            return fwk_path
        return ""

    @classmethod
    def get_cann_path(cls, profiler_path: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(profiler_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(profiler_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^PROF_\d+_\d+_[a-zA-Z]+", sub_dir):
                return sub_path
        return ""

    @classmethod
    def get_device_path(cls, cann_path: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(cann_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(cann_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^device_\d", sub_dir):
                return sub_path
        return ""

    @classmethod
    def get_profiler_path_list(cls, input_path: str) -> list:
        if not os.path.isdir(input_path):
            return []
        if cls.get_fwk_path(input_path) or cls.get_cann_path(input_path):
            return [input_path]
        sub_dirs = os.listdir(os.path.realpath(input_path))
        profiler_path_list = []
        for sub_dir in sub_dirs:
            sub_path = os.path.join(input_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            if cls.get_fwk_path(sub_path) or cls.get_cann_path(sub_path):
                profiler_path_list.append(sub_path)
        return profiler_path_list

    @classmethod
    def get_device_all_file_list_by_type(cls, profiler_path: str, summary_or_timeline: str) -> list:
        file_list = []
        _path = os.path.join(cls.get_device_path(profiler_path), summary_or_timeline)
        if not os.path.isdir(_path):
            return file_list
        sub_files = os.listdir(os.path.realpath(_path))
        if not sub_files:
            return file_list
        for sub_file in sub_files:
            file_list.append(os.path.join(_path, sub_file))
        return file_list
