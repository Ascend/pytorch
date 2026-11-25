import os
import re
from datetime import datetime, timezone

from ...utils._path_manager import PathManager
from ._profiler_config import ProfilerConfig
from .prof_common_func._cann_package_manager import CannPackageManager
from .prof_common_func._constant import Constant, print_info_msg, print_warn_msg
from .prof_common_func._file_manager import FileManager
from .prof_common_func._log import ProfilerLogger
from .prof_common_func._path_manager import ProfilerPathManager
from .prof_common_func._task_manager import ConcurrentTasksManager
from .prof_common_func._utils import no_exception_func
from .prof_config._parser_config import ParserConfig


__all__ = []


class ProfilingParser:
    def __init__(
        self, profiler_path: str, analysis_type: str, output_path: str, kwargs: dict
    ):
        self._profiler_path = profiler_path
        self._analysis_type = analysis_type
        self._output_path = output_path
        self._kwargs = kwargs
        self._start_time = datetime.now(tz=timezone.utc).astimezone()
        if analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            self._output_path = os.path.join(profiler_path, Constant.OUTPUT_DIR)
            PathManager.remove_path_safety(self._output_path)
            PathManager.make_dir_safety(self._output_path)
        self.logger = None

    @staticmethod
    def remove_cann_output(cann_path: str, remove_all: bool):
        """删除 PROF 目录下上一次解析/导出留下的中间产物，保留原始数据与日志。"""
        if not cann_path:
            return
        device_paths = ProfilerPathManager.get_device_path(cann_path)
        host_path = ProfilerPathManager.get_host_path(cann_path)
        rm_dirs = ["sqlite", "summary", "timeline"] if remove_all else ["sqlite"]
        for rm_dir in rm_dirs:
            for device_path in device_paths:
                PathManager.remove_path_safety(os.path.join(device_path, rm_dir))
            if host_path:
                PathManager.remove_path_safety(os.path.join(host_path, rm_dir))
        if not remove_all:
            return
        # mindstudio_profiler_log 与 msprof_analysis_*.log 均保留，便于问题定位
        for rm_dir in ["analyze", "mindstudio_profiler_output"]:
            PathManager.remove_path_safety(os.path.join(cann_path, rm_dir))

    @staticmethod
    def simplify_data(profiler_path: str, simplify_flag: bool):
        ProfilingParser.remove_cann_output(
            ProfilerPathManager.get_cann_path(profiler_path), simplify_flag
        )

    def update_export_type(self):
        export_type = self._kwargs.get("export_type", None)
        if export_type is not None:
            ProfilerConfig().export_type = export_type
        if Constant.Db not in ProfilerConfig().export_type:
            return
        if (
            self._analysis_type == Constant.EXPORT_CHROME_TRACE
            or self._analysis_type == Constant.EXPORT_STACK
        ):
            print_warn_msg(
                "The setting of type in experimental_config as db will be ignored while set export_chrome_trace or export_stacks"
            )
            ProfilerConfig().export_type = [Constant.Text]
            return
        if not ProfilerPathManager.get_cann_path(self._profiler_path):
            return

    def update_platform_analysis_dir(self):
        plt_dir_path = ProfilerPathManager.get_plt_dir_path(self._profiler_path)
        if not plt_dir_path:
            return
        if len(os.listdir(plt_dir_path)) == 0:
            PathManager.remove_path_safety(plt_dir_path)
            return
        tokens = os.path.basename(os.path.realpath(self._profiler_path)).split("_")
        if len(tokens) > 2 and tokens[2].isnumeric():
            ts_profiler_dir = tokens[2]
            new_plt_name = f"{Constant.PLATFORM_ANALYSIS_DIR}_{ts_profiler_dir}"
            future_plt_path = os.path.join(
                os.path.dirname(self._profiler_path), new_plt_name
            )
            if os.path.exists(future_plt_path):
                PathManager.remove_path_safety(future_plt_path)
            os.replace(plt_dir_path, future_plt_path)

    def delete_previous_cann_db_files(self):
        cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        if not cann_path:
            return
        if (
            self._analysis_type == Constant.TENSORBOARD_TRACE_HANDLER
            and Constant.Db in ProfilerConfig().export_type
        ):
            pattern = r"^msprof_\d+\.db$"
            for filename in os.listdir(cann_path):
                if re.match(pattern, filename) and os.path.isfile(
                    os.path.join(cann_path, filename)
                ):
                    PathManager.remove_file_safety(os.path.join(cann_path, filename))

    @no_exception_func()
    def analyse_profiling_data(self):
        ProfilerLogger.init(self._profiler_path, "ProfilingParser")
        self.logger = ProfilerLogger.get_instance()
        print_info_msg(
            f"Start parsing profiling data in "
            f"{'async' if self._kwargs.get('async_mode') else 'sync'} mode at: {self._profiler_path}"
        )
        ProfilerConfig().reset_is_load()
        ProfilerConfig().load_info(self._profiler_path)
        self.update_export_type()
        self.delete_previous_cann_db_files()
        try:
            self.run_parser()
        except Exception:
            self.logger.exception("Failed to parsing profiling data")
        if self._analysis_type == Constant.TENSORBOARD_TRACE_HANDLER:
            self.simplify_data(
                self._profiler_path, ProfilerConfig().data_simplification
            )
            self.update_platform_analysis_dir()
            self.generate_parser_done_file()
        end_time = datetime.now(tz=timezone.utc).astimezone()
        print_info_msg(
            f"All profiling data parsed in a total time of {end_time - self._start_time}"
        )

    def generate_parser_done_file(self):
        FileManager.create_empty_file(self._output_path, "analyse.done")

    def run_parser(self) -> list:
        param_dict = {
            "profiler_path": self._profiler_path,
            "output_path": self._output_path,
        }
        if self._kwargs:
            param_dict.update(self._kwargs)
        param_dict[Constant.EXPORT_TYPE] = ProfilerConfig().export_type
        param_dict[Constant.ACTIVITIES] = ProfilerConfig().activities
        parser_config = ParserConfig.ONLY_FWK_CONFIG
        cann_path = ProfilerPathManager.get_cann_path(self._profiler_path)
        if cann_path:
            # 解析前清理上一次的解析/导出残留，避免旧的 finish.txt、旧 csv/db
            # 影响本次导出的等待判断和解析结果
            ProfilingParser.remove_cann_output(cann_path, True)
            if ProfilerConfig().get_level() == "Level_none":
                parser_config = ParserConfig.LEVEL_NONE_CONFIG
            else:
                parser_config = ParserConfig.COMMON_CONFIG

        parser_list = []
        unique_parser_set = set()
        for export_type in set(ProfilerConfig().export_type):
            for parser in parser_config.get(export_type, {}).get(
                self._analysis_type, []
            ):
                if parser in unique_parser_set:
                    continue
                unique_parser_set.add(parser)
                parser_list.append(parser)

        # when cann package support default export db, add db parser to parser list
        if CannPackageManager.is_support_default_export_db():
            for parser in parser_config.get(Constant.Db, {}).get(
                self._analysis_type, []
            ):
                if parser in unique_parser_set:
                    continue
                unique_parser_set.add(parser)
                parser_list.append(parser)

        manager = ConcurrentTasksManager(progress_bar="cursor")
        for parser in parser_list:
            manager.add_task(
                parser(ParserConfig.PARSER_NAME_MAP.get(parser), param_dict)
            )
        manager.run()
