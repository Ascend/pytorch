import os
from datetime import datetime

from .prof_common_func.constant import Constant, print_info_msg, print_error_msg
from .prof_common_func.path_manager import ProfilerPathManager
from .prof_common_func.task_manager import ConcurrentTasksManager
from .prof_config.parser_config import ParserConfig
from .prof_parse.cann_file_parser import CANNFileParser
from .profiler_config import ProfilerConfig
from ...utils.path_manager import PathManager


class ProfilingParser:
    def __init__(self, profiler_path: str, analysis_type: str, output_path: str, kwargs: dict):
        self._profiler_path = profiler_path
        self._analysis_type = analysis_type
        self._output_path = output_path
        self._kwargs = kwargs
        self._start_time = datetime.utcnow()
        if analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            self._output_path = os.path.join(profiler_path, Constant.OUTPUT_DIR)
            PathManager.remove_path_safety(self._output_path)
            PathManager.make_dir_safety(self._output_path)

    def analyse_profiling_data(self):
        print_info_msg(f"Start parsing profiling data: {self._profiler_path}")
        try:
            self.run_parser()
        except Exception as err:
            print_error_msg(f"Failed to parsing profiling data. {err}")
        if self._analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            ProfilerPathManager.simplify_data(self._profiler_path, ProfilerConfig().data_simplification)
        end_time = datetime.utcnow()
        print_info_msg(f"All profiling data parsed in a total time of {end_time - self._start_time}")

    def run_parser(self) -> list:
        ProfilerConfig().load_info(self._profiler_path)
        param_dict = {"profiler_path": self._profiler_path, "output_path": self._output_path}
        if self._kwargs:
            param_dict.update(self._kwargs)
        if ProfilerPathManager.get_cann_path(self._profiler_path):
            CANNFileParser(self._profiler_path).del_summary_and_timeline_data()
            parser_list = ParserConfig.COMMON_CONFIG.get(self._analysis_type)
        else:
            parser_list = ParserConfig.ONLY_FWK_CONFIG.get(self._analysis_type)
        manager = ConcurrentTasksManager(progress_bar="cursor")
        for parser in parser_list:
            manager.add_task(parser(ParserConfig.PARSER_NAME_MAP.get(parser), param_dict))
        manager.run()
