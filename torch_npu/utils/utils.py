import os
import warnings
from warnings import _showwarnmsg_impl


def should_print_warning():
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
        if not should_print_warning() and npu_path in filename:
            return
        msg = warnings.WarningMessage(message, category, filename, lineno, file, line)
        _showwarnmsg_impl(msg)

    warnings.showwarning = npu_show_warning
