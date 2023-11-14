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
from datetime import timezone

from ....utils.path_manager import PathManager
from ..prof_common_func.constant import Constant, print_info_msg
from ..prof_common_func.global_var import GlobalVar
from ..prof_common_func.path_manager import ProfilerPathManager
from ..prof_config.view_parser_config import ViewParserConfig
from ..prof_parse.cann_file_parser import CANNFileParser
from ..profiler_config import ProfilerConfig


class ViewParserFactory:
    @classmethod
    def create_view_parser_and_run(cls, profiler_path: str, analysis_type: str, output_path: str, kwargs: dict):
        print_info_msg(f'Start parsing profiling data: {profiler_path}')
        start_time = datetime.datetime.now(tz=timezone.utc)
        ProfilerConfig().load_info(profiler_path)
        if ProfilerPathManager.get_cann_path(profiler_path):
            cann_file_parser = CANNFileParser(profiler_path)
            cann_file_parser.check_prof_data_size()
            CANNFileParser(profiler_path).export_cann_profiling(ProfilerConfig().data_simplification)
            end_time = datetime.datetime.now(tz=timezone.utc)
            print_info_msg(f'CANN profiling data parsed in a total time of {end_time - start_time}')
        GlobalVar.init(profiler_path)
        if analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            output_path = os.path.join(profiler_path, Constant.OUTPUT_DIR)
            PathManager.remove_path_safety(output_path)
            PathManager.make_dir_safety(output_path)
        for parser in ViewParserConfig.CONFIG_DICT.get(analysis_type):
            parser(profiler_path).generate_view(output_path, **kwargs)
        GlobalVar.torch_op_tree_node = []
        if ProfilerConfig().data_simplification:
            ProfilerPathManager.simplify_data(profiler_path)
        end_time = datetime.datetime.now(tz=timezone.utc)
        print_info_msg(f'All profiling data parsed in a total time of {end_time - start_time}')
