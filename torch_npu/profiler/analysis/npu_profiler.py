import os
from multiprocessing import Process, Pool

from .prof_common_func.constant import Constant
from .prof_view.view_parser_factory import ViewParserFactory
from .prof_common_func.path_manager import ProfilerPathManager
from ...utils.path_manager import PathManager


class NpuProfiler:

    @classmethod
    def analyse(cls, input_path: str, analysis_type: str = Constant.TENSORBOARD_TRACE_HANDLER, output_path: str = None,
                **kwargs):
        input_path = ProfilerPathManager.get_realpath(input_path)
        cls._check_input_path(input_path)
        profiler_path_list = ProfilerPathManager.get_profiler_path_list(input_path)
        if not profiler_path_list:
            return
        # 多个profiler用多进程处理
        pool = Pool(processes=os.cpu_count() // 2)
        for profiler_path in profiler_path_list:
            PathManager.check_directory_path_writeable(profiler_path)
            pool.apply_async(ViewParserFactory.create_view_parser_and_run,
                             (profiler_path, analysis_type, output_path, kwargs))
        pool.close()
        pool.join()

    @classmethod
    def _check_input_path(cls, path: str):
        PathManager.check_input_directory_path(path)
        PathManager.check_path_owner_consistent(path)
