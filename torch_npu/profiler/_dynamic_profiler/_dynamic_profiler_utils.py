import os
import socket
import logging
from enum import Enum
from logging.handlers import RotatingFileHandler
import torch

from ...utils._path_manager import PathManager
from ..analysis.prof_common_func._constant import print_info_msg
from ..analysis.prof_common_func._constant import print_warn_msg
from ..analysis.prof_common_func._constant import print_error_msg


class DynamicProfilerUtils:
    class DynamicProfilerConfigModel(Enum):
        CFG_CONFIG = 0  # 配置文件使能方式
        DYNO_CONFIG = 1  # Dynolog使能方式

    class LoggerLevelEnum(Enum):
        INFO = 0
        WARNING = 1
        ERROR = 2

    DYNAMIC_PROFILER_MODEL = DynamicProfilerConfigModel.CFG_CONFIG
    LOGGER = logging.getLogger("DynamicProfiler")
    LOGGER_MONITOR = logging.getLogger("DynamicProfilerMonitor")
    DYNAMIC_LOG_TO_FILE_MAP = {
        LoggerLevelEnum.INFO: LOGGER.info,
        LoggerLevelEnum.WARNING: LOGGER.warning,
        LoggerLevelEnum.ERROR: LOGGER.error
    }
    DYNAMIC_MONITOR_LOG_TO_FILE_MAP = {
        LoggerLevelEnum.INFO: LOGGER_MONITOR.info,
        LoggerLevelEnum.WARNING: LOGGER_MONITOR.warning,
        LoggerLevelEnum.ERROR: LOGGER_MONITOR.error
    }

    DYNAMIC_LOG_TO_STDOUT_MAP = {
        LoggerLevelEnum.INFO: print_info_msg,
        LoggerLevelEnum.WARNING: print_warn_msg,
        LoggerLevelEnum.ERROR: print_error_msg
    }

    CFG_CONFIG_PATH = None
    CFG_BUFFER_SIZE = 1024 * 1024
    POLL_INTERVAL = 2

    @classmethod
    def init_logger(cls, is_monitor_process: bool = False):
        logger_ = cls.LOGGER_MONITOR if is_monitor_process else cls.LOGGER
        path = cls.CFG_CONFIG_PATH
        path = os.path.join(path, 'log')
        if not os.path.exists(path):
            PathManager.make_dir_safety(path)
        worker_name = "{}".format(socket.gethostname())
        log_name = "dp_{}_{}_rank_{}.log".format(worker_name, os.getpid(), cls.get_rank_id())
        if is_monitor_process:
            log_name = "monitor_" + log_name
        log_file = os.path.join(path, log_name)
        if not os.path.exists(log_file):
            PathManager.create_file_safety(log_file)
        handler = RotatingFileHandler(filename=log_file, maxBytes=1024 * 200, backupCount=1)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(process)d] %(filename)s: %(message)s")
        handler.setFormatter(formatter)
        logger_.setLevel(logging.DEBUG)
        logger_.addHandler(handler)

    @classmethod
    def is_dyno_model(cls):
        if DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL == DynamicProfilerUtils.DynamicProfilerConfigModel.CFG_CONFIG:
            return False
        return True

    @classmethod
    def out_log(cls, message: str, level: LoggerLevelEnum = LoggerLevelEnum.INFO, is_monitor_process: bool = False):
        if not cls.is_dyno_model():
            if not is_monitor_process:
                cls.DYNAMIC_LOG_TO_FILE_MAP[level](message)
            else:
                cls.DYNAMIC_MONITOR_LOG_TO_FILE_MAP[level](message)
        else:
            cls.stdout_log(message, level)

    @classmethod
    def stdout_log(cls, message: str, level: LoggerLevelEnum = LoggerLevelEnum.INFO):
        cls.DYNAMIC_LOG_TO_STDOUT_MAP[level](message)

    @staticmethod
    def get_rank_id() -> int:
        try:
            rank_id = os.environ.get('RANK')
            if rank_id is None and torch.distributed.is_available() and torch.distributed.is_initialized():
                rank_id = torch.distributed.get_rank()
            if not isinstance(rank_id, int):
                rank_id = int(rank_id)
        except Exception as ex:
            print_warn_msg(f"Get rank id  {str(ex)}, rank_id will be set to -1 !")
            rank_id = -1

        return rank_id

    @staticmethod
    def dyno_str_to_json(res: str):
        try:
            res_dict = {}
            pairs = str(res).split("\n")
            char_equal = '='
            for pair in pairs:
                str_split = pair.split(char_equal)
                if len(str_split) == 2:
                    res_dict[str_split[0].strip()] = str_split[1].strip()
        except Exception as ex:
            print_warn_msg(f"Dyno request response is not valid, occur error {ex}!")
            res_dict = {}

        return res_dict

    @staticmethod
    def parse_str_params_to_list(params):
        if params is None or params == 'None':
            return []
        return [item.strip() for item in params.split(',')]
