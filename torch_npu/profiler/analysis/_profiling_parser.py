import os
import re
from datetime import datetime

from .prof_common_func._constant import Constant, print_info_msg, print_error_msg, print_warn_msg
from .prof_common_func._cann_package_manager import CannPackageManager
from .prof_common_func._path_manager import ProfilerPathManager
from .prof_common_func._task_manager import ConcurrentTasksManager
from .prof_common_func._log import ProfilerLogger
from .prof_config._parser_config import ParserConfig
from .prof_parse._cann_file_parser import CANNFileParser
from ._profiler_config import ProfilerConfig
from ...utils._path_manager import PathManager

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
        self.logger = None

    @staticmethod
    def simplify_data(profiler_path: str, simplify_flag: bool):
        cann_path = ProfilerPathManager.get_cann_path(profiler_path)
        device_path = ProfilerPathManager.get_device_path(cann_path)
        host_path = ProfilerPathManager.get_host_path(cann_path)
        rm_dirs = ['sqlite', 'summary', 'timeline'] if simplify_flag else ['sqlite']
        for rm_dir in rm_dirs:
            if device_path:
                target_path = os.path.join(device_path, rm_dir)
                PathManager.remove_path_safety(target_path)
            if host_path:
                target_path = os.path.join(host_path, rm_dir)
                PathManager.remove_path_safety(target_path)
        if simplify_flag:
            if not cann_path:
                return
            cann_rm_dirs = ['analyze', 'mindstudio_profiler_log', 'mindstudio_profiler_output']
            for cann_rm_dir in cann_rm_dirs:
                target_path = os.path.join(cann_path, cann_rm_dir)
                PathManager.remove_path_safety(target_path)
            log_patten = r'msprof_analysis_\d+\.log$'
            for cann_file in os.listdir(cann_path):
                file_path = os.path.join(cann_path, cann_file)
                if not os.path.isfile(file_path):
                    continue
                if re.match(log_patten, cann_file):
                    PathManager.remove_file_safety(file_path)

    def update_export_type(self):
        export_type = self._kwargs.get('export_type', None)
        if export_type is not None:
            ProfilerConfig().export_type = export_type
        if Constant.Db not in ProfilerConfig().export_type:
            return
        if self._analysis_type == Constant.EXPORT_CHROME_TRACE or self._analysis_type == Constant.EXPORT_STACK:
            print_warn_msg("The setting of type in experimental_config as db will be ignored while set export_chrome_trace or export_stacks")
            ProfilerConfig().export_type = [Constant.Text]
            return
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            return
        if not CannPackageManager.is_support_export_db():
            raise RuntimeError("Current CANN package version does not support export db. "
                               "If you want to export db, you can install supported CANN package version.")

    def delete_previous_cann_db_files(self):
        cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        if not cann_path:
            return
        if self._analysis_type == Constant.TENSORBOARD_TRACE_HANDLER and Constant.Db in ProfilerConfig().export_type:
            patten = r'^msprof_\d+\.db$'
            for filename in os.listdir(cann_path):
                if re.match(patten, filename) and os.path.isfile(os.path.join(cann_path, filename)):
                    PathManager.remove_file_safety(os.path.join(cann_path, filename))

    def analyse_profiling_data(self):
        ProfilerLogger.init(self._profiler_path, "ProfilingParser")
        self.logger = ProfilerLogger.get_instance()
        print_info_msg(f"Start parsing profiling data: {self._profiler_path}")
        ProfilerConfig().load_info(self._profiler_path)
        self.update_export_type()
        self.delete_previous_cann_db_files()
        try:
            self.run_parser()
        except Exception as err:
            print_error_msg(f"Failed to parsing profiling data. {err}")
            self.logger.error("Failed to parsing profiling data, error: %s", str(err), exc_info=True)
        if self._analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            self.simplify_data(self._profiler_path, ProfilerConfig().data_simplification)
        end_time = datetime.utcnow()
        print_info_msg(f"All profiling data parsed in a total time of {end_time - self._start_time}")

    def run_parser(self) -> list:
        param_dict = {"profiler_path": self._profiler_path, "output_path": self._output_path}
        if self._kwargs:
            param_dict.update(self._kwargs)
        param_dict[Constant.EXPORT_TYPE] = ProfilerConfig().export_type

        parser_config = ParserConfig.ONLY_FWK_CONFIG
        if ProfilerPathManager.get_cann_path(self._profiler_path):
            CANNFileParser(self._profiler_path).del_summary_and_timeline_data()
            CANNFileParser(self._profiler_path).del_output_path_data()
            if ProfilerConfig().get_level() == "Level_none":
                parser_config = ParserConfig.LEVEL_NONE_CONFIG
            else:
                parser_config = ParserConfig.COMMON_CONFIG

        parser_list = []
        unique_parser_set = set()
        for export_type in set(ProfilerConfig().export_type):
            for parser in parser_config.get(export_type).get(self._analysis_type):
                if parser in unique_parser_set:
                    continue
                unique_parser_set.add(parser)
                parser_list.append(parser)

        manager = ConcurrentTasksManager(progress_bar="cursor")
        for parser in parser_list:
            manager.add_task(parser(ParserConfig.PARSER_NAME_MAP.get(parser), param_dict))
        manager.run()
