import os
from multiprocessing import Process

from .prof_common_func.constant import Constant
from .prof_view.view_parser_factory import ViewParserFactory
from .prof_common_func.path_manager import PathManager
from ...utils.secure_path_manager import SecurePathManager


class NpuProfiler:

    @classmethod
    def analyse(cls, input_path: str, analysis_type: str = Constant.TENSORBOARD_TRACE_HANDLER, output_path: str = None,
                **kwargs):
        input_path = PathManager.get_realpath(input_path)
        cls._check_input_path(input_path)
        profiler_path_list = PathManager.get_profiler_path_list(input_path)
        if not profiler_path_list:
            return
        # 多个profiler用多进程处理
        process_list = []
        for profiler_path in profiler_path_list:
            SecurePathManager.check_directory_path_writeable(profiler_path)
            process = Process(target=ViewParserFactory.create_view_parser_and_run,
                              args=(profiler_path, analysis_type, output_path, kwargs))
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

    @classmethod
    def _check_input_path(cls, path: str):
        SecurePathManager.check_input_directory_path(path)
        SecurePathManager.check_path_owner_consistent(path)
