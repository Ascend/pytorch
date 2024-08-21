import os
import socket
import logging
from logging.handlers import RotatingFileHandler
from ...utils.path_manager import PathManager

logger = logging.getLogger("DynamicProfiler")
logger_monitor = logging.getLogger("DynamicProfilerMonitor")


def init_logger(logger_: logging.Logger, path: str, is_monitor_process=False):
    path = os.path.join(path, 'log')
    if not os.path.exists(path):
        PathManager.make_dir_safety(path)
    worker_name = "{}".format(socket.gethostname())
    log_name = "dp_{}_{}.log".format(worker_name, os.getpid())
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
