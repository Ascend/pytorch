import os
import sys
import time
import warnings
from warnings import _showwarnmsg_impl

__all__ = []


class _LogLevel:
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


def _print_log(level: str, msg: str):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getpid()
    print(f"{current_time}({pid})-[{level}] {msg}")


def _print_info_log(info_msg: str):
    _print_log(_LogLevel.INFO, info_msg)


def _print_warn_log(warn_msg: str):
    _print_log(_LogLevel.WARNING, warn_msg)


def _print_error_log(error_msg: str):
    _print_log(_LogLevel.ERROR, error_msg)


def _should_print_warning():
    disabled_warning = os.environ.get("TORCH_NPU_DISABLED_WARNING", "0")
    if disabled_warning == "1":
        return False
    rank = os.environ.get("RANK", None)
    if rank is None or rank == "0":
        return True
    return False


def _apply_npu_show_warning():
    def npu_show_warning(message, category, filename, lineno, file=None, line=None):
        npu_path = os.path.dirname(os.path.dirname(__file__))
        if not _should_print_warning() and npu_path in filename:
            return
        msg = warnings.WarningMessage(message, category, filename, lineno, file, line)
        _showwarnmsg_impl(msg)

    warnings.showwarning = npu_show_warning


def _is_interactive_command_line():
    # check whether it is standard python interactive environment
    if hasattr(sys, 'ps1'):
        return True

    # check whether it is IPython or Jupyter environment
    try:
        __IPYTHON__  # noqa: F821
        return True
    except NameError:
        pass

    # check whether it is Python REPL mode
    if sys.flags.interactive:
        return True

    # check whether it is notebook mode
    if 'ipykernel' in sys.modules:
        return True

    return False
