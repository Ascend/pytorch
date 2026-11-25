import os
from datetime import datetime, timezone

from torch_npu.utils._path_manager import PathManager

from ...prof_common_func._constant import print_warn_msg
from ...prof_common_func._log import ProfilerLogger

__all__ = []

# 与 ProfilerLogger 写日志的目录保持一致，data_simplification 开启时也会保留
COMMAND_LOG_PREFIX = "msprof"


def build_command_log_path(profiler_path: str, mode: str) -> str:
    """按 logs 下 profiler_<时间戳>_<pid>_<名字>.log 的习惯，拼出 msprof 打屏的落盘路径。"""
    timestamp = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d%H%M%S")
    file_name = f"{COMMAND_LOG_PREFIX}_{mode}_{timestamp}_{os.getpid()}.log"
    return os.path.join(profiler_path, ProfilerLogger.DEFAULT_LOG_DIR, file_name)


def save_command_output(log_path: str, cmd_list: list, completed_process) -> None:
    """把 msprof 子进程的原始打屏追加到日志文件，落盘失败不影响解析主流程。"""
    try:
        PathManager.make_dir_safety(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as file:
            file.write("[command] " + " ".join(str(arg) for arg in cmd_list) + "\n")
            file.write(f"[returncode] {completed_process.returncode}\n")
            file.write(f"[stdout]\n{_to_text(completed_process.stdout)}\n")
            file.write(f"[stderr]\n{_to_text(completed_process.stderr)}\n\n")
    except Exception as err:
        print_warn_msg(f"Failed to save msprof command output to {log_path}, error: {err}")


def _to_text(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)
