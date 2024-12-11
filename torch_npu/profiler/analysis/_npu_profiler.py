import multiprocessing

from .prof_common_func._constant import Constant, print_error_msg
from .prof_common_func._path_manager import ProfilerPathManager
from .prof_common_func._multi_process_pool import MultiProcessPool
from ._profiling_parser import ProfilingParser
from ...utils._path_manager import PathManager

__all__ = []


class NpuProfiler:

    @classmethod
    def analyse(cls, input_path: str, analysis_type: str = Constant.TENSORBOARD_TRACE_HANDLER, output_path: str = None,
                **kwargs):
        input_path = ProfilerPathManager.get_realpath(input_path)
        cls._check_input_path(input_path)
        profiler_path_list = ProfilerPathManager.get_profiler_path_list(input_path)
        if not profiler_path_list:
            return
        if multiprocessing.current_process().daemon:
            message = "The profiling data cannot be parsed during the daemon process, it is recommended that " \
                      "you use an offline parsing interface to parse the collected data. \n" \
                      "For example: \nfrom torch_npu.profiler.profiler import analyse\n" \
                      "analyse(\"profiling_data_path\")"
            print_error_msg(message)
            return

        # 多profiling数据的解析
        multiprocessing.set_start_method("fork", force=True)
        process_number = min(kwargs.get("max_process_number", Constant.DEFAULT_PROCESS_NUMBER),
                             len(profiler_path_list))
        MultiProcessPool().init_pool(process_number)

        for profiler_path in profiler_path_list:
            PathManager.check_directory_path_writeable(profiler_path)
            profiling_parser = ProfilingParser(profiler_path, analysis_type, output_path, kwargs)
            MultiProcessPool().submit_task(profiling_parser.analyse_profiling_data)
        if not kwargs.get("async_mode", False):
            MultiProcessPool().close_pool()

    @classmethod
    def _check_input_path(cls, path: str):
        PathManager.check_input_directory_path(path)
        PathManager.check_path_owner_consistent(path)
