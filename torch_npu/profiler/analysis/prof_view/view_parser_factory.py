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
import datetime
import os
import shutil
import threading
import time

from ..prof_common_func.constant import Constant
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.global_var import GlobalVar
from ..prof_common_func.path_manager import PathManager
from ..prof_config.view_parser_config import ViewParserConfig
from ..prof_parse.cann_file_parser import CANNFileParser
from ..profiler_config import ProfilerConfig


class ViewParserFactory:
    @classmethod
    def create_view_parser_and_run(cls, profiler_path: str, analysis_type: str, output_path: str):
        print(f"[INFO] [{os.getpid()}] profiler.py: Start parsing profiling data.")
        ProfilerConfig().load_info(profiler_path)
        cann_file_parser = CANNFileParser(profiler_path)
        cann_file_parser.check_prof_data_size()
        start_time = datetime.datetime.now()
        CANNFileParser(profiler_path).export_cann_profiling(ProfilerConfig().data_simplification)
        end_time = datetime.datetime.now()
        print(
            f"[INFO] [{os.getpid()}] profiler.py: CANN profiling data parsed in a total time of {end_time - start_time}")
        GlobalVar.init(profiler_path)
        if analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            output_path = os.path.join(profiler_path, Constant.OUTPUT_DIR)
            FileManager.remove_and_make_output_dir(output_path)
        for parser in ViewParserConfig.CONFIG_DICT.get(analysis_type):
            parser(profiler_path).generate_view(output_path)
        cls.simplify_data(profiler_path)
        end_time = datetime.datetime.now()
        print(
            f"[INFO] [{os.getpid()}] profiler.py: All profiling data parsed in a total time of {end_time - start_time}")

    @classmethod
    def simplify_data(cls, profiler_path: str):
        if not ProfilerConfig().data_simplification:
            return
        target_path = os.path.join(profiler_path, Constant.FRAMEWORK_DIR)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
