import os
import logging
import torch._logging._internal
from torch_npu import _C


def _set_logs():
    if os.environ.get('TORCH_LOGS', None) is not None or os.environ.get('TORCH_NPU_LOGS', None) is not None:
        return

    _C._logging._LogContext.GetInstance().setLogs(torch._logging._internal.log_state.log_qname_to_level)


def _trigger_set_logs_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        _set_logs()
        return result
    return wrapper


def _logging_patch():
    torch._logging.set_logs = _trigger_set_logs_decorator(torch._logging.set_logs)


def _add_logging_module():
    torch._logging._internal.register_log("memory", "torch_npu.memory")
    torch._logging._internal.register_log("delivery", "torch_npu.delivery")
