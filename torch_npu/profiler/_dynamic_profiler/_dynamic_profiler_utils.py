import os
import socket
import logging
from logging.handlers import RotatingFileHandler
import torch
from ...utils._path_manager import PathManager

logger = logging.getLogger("DynamicProfiler")
logger_monitor = logging.getLogger("DynamicProfilerMonitor")


def init_logger(logger_: logging.Logger, path: str, is_monitor_process: bool = False):
    path = os.path.join(path, 'log')
    if not os.path.exists(path):
        PathManager.make_dir_safety(path)
    worker_name = "{}".format(socket.gethostname())
    log_name = "dp_{}_{}_rank_{}.log".format(worker_name, os.getpid(), _get_rank_id())
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


def _get_rank_id() -> int:
    try:
        rank_id = os.environ.get('RANK')
        if rank_id is None and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_id = torch.distributed.get_rank()
        if not isinstance(rank_id, int):
            rank_id = int(rank_id)
    except Exception as ex:
        logger.warning("Get rank id  %s, rank_id will be set to -1 !", str(ex))
        rank_id = -1

    return rank_id


def _get_device_id() -> int:
    try:
        device_id = os.environ.get('LOCAL_RANK')
        if not isinstance(device_id, int):
            device_id = int(device_id)
    except Exception as ex:
        logger.warning("Get device id  %s, device_id will be set to -1 !", str(ex))
        device_id = -1

    return device_id
