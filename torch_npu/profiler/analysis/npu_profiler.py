import multiprocessing
import os
from multiprocessing.pool import Pool

from .prof_common_func.constant import Constant
from .prof_common_func.path_manager import ProfilerPathManager
from .prof_common_func.prof_process import ProfProcess
from .profiling_parser import ProfilingParser
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
        # 多profiling数据的解析
        multiprocessing.set_start_method("fork", force=True)
        pool = ProfProcessPool(processes=os.cpu_count() // 2)
        for profiler_path in profiler_path_list:
            PathManager.check_directory_path_writeable(profiler_path)
            profiling_parser = ProfilingParser(profiler_path, analysis_type, output_path, kwargs)
            pool.apply_async(profiling_parser.analyse_profiling_data)
        pool.close()
        pool.join()

    @classmethod
    def _check_input_path(cls, path: str):
        PathManager.check_input_directory_path(path)
        PathManager.check_path_owner_consistent(path)


class ProfContext(type(multiprocessing.get_context())):
    Process = ProfProcess


class ProfProcessPool(Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = ProfContext()
        super(ProfProcessPool, self).__init__(*args, **kwargs)
