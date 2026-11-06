import os
import re
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

    class ProfilerStatus(Enum):
        UNINITIALIZED = -1
        IDLE = 0
        RUNNING = 1
        READY = 2

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

    # 状态上报信息
    PROFILER_STATUS = "profiler_status"
    CURRENT_STEP = "current_step"
    START_STEP = "start_step"
    STOP_STEP = "stop_step"
    TRACE_ID = "trace_id"
    QUEUE_DEPTH = "queue_depth"
    REJECTED = "rejected"
    REPORT_INTERVAL = 1.0

    RECORD_SEPARATOR = '\x1e'
    BATCH_COUNT_MARKER = 'DYNO_BATCH_COUNT'
    BATCH_LEN_MARKER = 'DYNO_BATCH_LEN'
    BATCH_SUM_MARKER = 'DYNO_BATCH_SUM'

    @classmethod
    def init_logger(cls, is_monitor_process: bool = False, log_path: str = None):
        logger_ = cls.LOGGER_MONITOR if is_monitor_process else cls.LOGGER
        path = os.path.join(log_path, 'log')
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
        res_dict = {}
        try:
            for pair in str(res).split("\n"):
                kv = pair.split("=")
                if len(kv) == 2:
                    res_dict[kv[0].strip()] = kv[1].strip()
        except Exception as ex:
            print_warn_msg(f"Dyno request response is not valid, occur error {ex}!")
        return res_dict

    @staticmethod
    def _parse_marker(head: str, marker: str):
        matches = re.findall(r'^' + marker + r'\s*=\s*(\S+)\s*$', head, re.MULTILINE)
        return matches[-1] if matches else None

    @staticmethod
    def _parse_int_marker(head: str, marker: str):
        value = DynamicProfilerUtils._parse_marker(head, marker)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fnv1a64(data: bytes) -> int:
        value = 0xcbf29ce484222325
        for byte in data:
            value ^= byte
            value = (value * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return value

    @staticmethod
    def parse_dyno_configs(record: str) -> list:
        if not record:
            return []
        head, _, payload = record.partition(DynamicProfilerUtils.RECORD_SEPARATOR)
        count = DynamicProfilerUtils._parse_int_marker(
            head, DynamicProfilerUtils.BATCH_COUNT_MARKER)
        payload_len = DynamicProfilerUtils._parse_int_marker(
            head, DynamicProfilerUtils.BATCH_LEN_MARKER)
        checksum = DynamicProfilerUtils._parse_marker(
            head, DynamicProfilerUtils.BATCH_SUM_MARKER)
        if count is None or payload_len is None or checksum is None:
            legacy_cfg = DynamicProfilerUtils.dyno_str_to_json(record)
            if legacy_cfg:
                return [legacy_cfg]
            print_error_msg("Dyno config frame is missing batch metadata, drop the frame!")
            return []
        payload_bytes = payload.encode('utf-8')
        if payload_len != len(payload_bytes):
            print_error_msg(f"Dyno config frame length mismatch, expect {payload_len}, but got {len(payload_bytes)}!")
            return []
        expect_sum = '{:016x}'.format(DynamicProfilerUtils._fnv1a64(payload_bytes))
        if checksum.lower() != expect_sum:
            print_error_msg(f"Dyno config frame checksum mismatch, expect {expect_sum}, but got {checksum}!")
            return []
        parts = payload.split(DynamicProfilerUtils.RECORD_SEPARATOR)
        if count != len(parts):
            print_error_msg(f"Dyno config frame count mismatch, expect {count}, but got {len(parts)}!")
            return []
        configs = []
        for part in parts:
            cfg = DynamicProfilerUtils.dyno_str_to_json(part)
            if not cfg:
                print_error_msg("Dyno config frame contains an empty config, drop the batch!")
                return []
            configs.append(cfg)
        return configs

    @staticmethod
    def parse_str_params_to_list(params):
        if params is None or params == 'None':
            return []
        return [item.strip() for item in params.split(',')]
