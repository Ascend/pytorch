import os
import time


class LogLevel:
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


def _print_log(level: str, msg: str):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getpid()
    print(f"{current_time}({pid})-[{level}] {msg}")


def print_info_log(info_msg: str):
    _print_log(LogLevel.INFO, info_msg)


def print_warn_log(warn_msg: str):
    _print_log(LogLevel.WARNING, warn_msg)


def print_error_log(error_msg: str):
    _print_log(LogLevel.ERROR, error_msg)
