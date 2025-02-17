__all__ = []

import os
import re
import logging
import torch._logging._internal
from torch_npu import _C
from ._internal import _logging_patch, _add_logging_module


_C._logging_init()
_logging_patch()
_add_logging_module()


def _update_log_state_from_env():
    log_setting = os.environ.get("TORCH_NPU_LOGS", None)
    if log_setting is not None:
        torch._logging._internal.LOG_ENV_VAR = "TORCH_NPU_LOGS"
        torch._logging._internal._init_logs()
        _C._logging._LogContext.GetInstance().setLogs(torch._logging._internal.log_state.log_qname_to_level)
    elif os.environ.get("TORCH_LOGS", None) is not None:
        _C._logging._LogContext.GetInstance().setLogs(torch._logging._internal.log_state.log_qname_to_level)

_update_log_state_from_env()
