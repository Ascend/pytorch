import os
import re
from datetime import datetime

from .prof_common_func.constant import Constant, print_info_msg, print_error_msg, print_warn_msg
from .prof_common_func.cann_package_manager import CannPackageManager
from .prof_common_func.path_manager import ProfilerPathManager
from .prof_common_func.task_manager import ConcurrentTasksManager
from .prof_config.parser_config import ParserConfig
from .prof_parse.cann_file_parser import CANNFileParser
from .profiler_config import ProfilerConfig
from ...utils.path_manager import PathManager

__all__ = []


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

    def update_export_type(self):
        if ProfilerConfig().export_type == Constant.Text:
            return
        if self._analysis_type == Constant.EXPORT_CHROME_TRACE or self._analysis_type == Constant.EXPORT_STACK:
            print_warn_msg("The setting of type in experimental_config as db will be ignored while set export_chrome_trace or export_stacks")
            ProfilerConfig().export_type = Constant.Text
            return
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            return
        if not CannPackageManager.cann_package_support_export_db():
            raise RuntimeError("Current CANN package version does not support export db. "
                               "If you want to export db, you can install supported CANN package version.")

    def delete_previous_cann_db_files(self):
        cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        if not cann_path:
            return
        if self._analysis_type == Constant.TENSORBOARD_TRACE_HANDLER and ProfilerConfig().export_type == Constant.Db:
            patten = r'^msprof_\d+\.db$'
            for filename in os.listdir(cann_path):
                if re.match(patten, filename) and os.path.isfile(os.path.join(cann_path, filename)):
                    PathManager.remove_file_safety(os.path.join(cann_path, filename))

    def analyse_profiling_data(self):
        print_info_msg(f"Start parsing profiling data: {self._profiler_path}")
        ProfilerConfig().load_info(self._profiler_path)
        self.update_export_type()
        self.delete_previous_cann_db_files()
        try:
            self.run_parser()
        except Exception as err:
            print_error_msg(f"Failed to parsing profiling data. {err}")
        if self._analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            ProfilerPathManager.simplify_data(self._profiler_path, ProfilerConfig().data_simplification)
        end_time = datetime.utcnow()
        print_info_msg(f"All profiling data parsed in a total time of {end_time - self._start_time}")

    def run_parser(self) -> list:
        param_dict = {"profiler_path": self._profiler_path, "output_path": self._output_path}
        if self._kwargs:
            param_dict.update(self._kwargs)
        if ProfilerPathManager.get_cann_path(self._profiler_path):
            CANNFileParser(self._profiler_path).del_summary_and_timeline_data()
            CANNFileParser(self._profiler_path).del_output_path_data() 
            if ProfilerConfig().get_level() == "Level_none":
                parser_list = ParserConfig.LEVEL_NONE_CONFIG.get(ProfilerConfig().export_type).get(self._analysis_type)
            else:
                parser_list = ParserConfig.COMMON_CONFIG.get(ProfilerConfig().export_type).get(self._analysis_type)
        else:
            parser_list = ParserConfig.ONLY_FWK_CONFIG.get(ProfilerConfig().export_type).get(self._analysis_type)
        manager = ConcurrentTasksManager(progress_bar="cursor")
        for parser in parser_list:
            manager.add_task(parser(ParserConfig.PARSER_NAME_MAP.get(parser), param_dict))
        manager.run()
