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
        cls.simplify_data(profiler_path)
        end_time = datetime.datetime.now(tz=timezone.utc)
        print_info_msg(f'All profiling data parsed in a total time of {end_time - start_time}')

    @classmethod
    def simplify_data(cls, profiler_path: str):
        if not ProfilerConfig().data_simplification:
            return
        target_path = os.path.join(profiler_path, Constant.FRAMEWORK_DIR)
        PathManager.remove_path_safety(target_path)
