import os
import logging
import torch._logging._internal
from torch_npu import _C


def _set_logs():
    """
    Propagate the results torch._logging.set_logs to the C++ layer.

    .. note:: The ``TORCH_LOGS`` or ``TORCH_NPU_LOGS`` environment variable has complete precedence
        over this function, so if it was set, this function does nothing.

    """

    # ignore if env var is set
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
    torch._logging._internal.register_log("dispatch", "torch_npu.dispatch")
    torch._logging._internal.register_log("dispatch_time", "torch_npu.dispatch.time")
    torch._logging._internal.register_log("silent", "torch_npu.silent_check")
    torch._logging._internal.register_log("recovery", "torch_npu.recovery")
    torch._logging._internal.register_log("op_plugin", "torch_npu.op_plugin")
    torch._logging._internal.register_log("shmem", "torch_npu.symmetric_memory")
    torch._logging._internal.register_log("env", "torch_npu.env")
    torch._logging._internal.register_log("acl", "torch_npu.acl")
    torch._logging._internal.register_log("aclgraph", "torch_npu.aclgraph")
