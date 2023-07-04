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

from ..prof_common_func.constant import Constant
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.global_var import GlobalVar
from ..prof_config.view_parser_config import ViewParserConfig
from ..prof_parse.cann_file_parser import CANNFileParser
from ..level_config import LevelConfig


class ViewParserFactory:
    @classmethod
    def create_view_parser_and_run(cls, profiler_path: str, output_path: str, level_config: dict = None):
        CANNFileParser(profiler_path).export_cann_profiling()
        GlobalVar.init(profiler_path)
        LevelConfig().load_info(level_config)
        if output_path:
            for parser in ViewParserConfig.CONFIG_DICT.get(Constant.EXPORT_CHROME_TRACE):
                parser(profiler_path).generate_view(output_path)
        else:
            FileManager.remove_and_make_output_dir(profiler_path)
            for parser in ViewParserConfig.CONFIG_DICT.get(Constant.TENSORBOARD_TRACE_HABDLER):
                parser(profiler_path).generate_view()
