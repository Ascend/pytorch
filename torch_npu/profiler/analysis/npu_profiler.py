from multiprocessing import Process

from .prof_view.view_parser_factory import ViewParserFactory
from .prof_common_func.path_manager import PathManager


class NpuProfiler:

    @classmethod
    def analyse(cls, input_path: str, level_config: dict = None, output_path: str = None):
        profiler_path_list = PathManager.get_profiler_path_list(input_path)
        if not profiler_path_list:
            return
        # 多个profiler用多进程处理
        process_list = []
        for profiler_path in profiler_path_list:
            process = Process(target=ViewParserFactory.create_view_parser_and_run,
                              args=(profiler_path, output_path, level_config))
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()
