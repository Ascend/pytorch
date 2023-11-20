import os
import socket
from datetime import datetime

from ..utils.path_manager import PathManager
from .analysis.prof_common_func.constant import Constant
from .analysis.prof_common_func.singleton import Singleton
from .analysis.prof_common_func.constant import print_warn_msg


@Singleton
class ProfManager:

    def __init__(self):
        self._prof_path = None
        self._worker_name = None
        self._dir_path = None
        self.is_prof_inited = False

    def init(self, worker_name: str = None, dir_path: str = None) -> None:
        if worker_name and isinstance(worker_name, str):
            self._worker_name = worker_name
        else:
            print_warn_msg("Input work_name must be str, use default value instead.")
            self._worker_name = None
        if dir_path and isinstance(dir_path, str):
            self._dir_path = dir_path
        else:
            print_warn_msg("Input dir_path must be str, use default value instead.")
            self._dir_path = None

    def create_prof_dir(self):
        if not self._dir_path:
            dir_path = os.getenv(Constant.ASCEND_WORK_PATH, default=None)
            dir_path = os.path.join(os.path.abspath(dir_path), Constant.PROFILING_WORK_PATH) if dir_path else os.getcwd()
        else:
            dir_path = self._dir_path
        if not self._worker_name:
            worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        else:
            worker_name = self._worker_name
        span_name = "{}_{}_ascend_pt".format(worker_name, datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:-3])
        self._prof_path = os.path.join(dir_path, span_name)
        PathManager.check_input_directory_path(self._prof_path)
        PathManager.make_dir_safety(self._prof_path)
        PathManager.check_directory_path_writeable(self._prof_path)
        self.is_prof_inited = True

    def get_prof_dir(self) -> str:
        return self._prof_path
