import os
import sys
import re
import time
from enum import Enum
from torch_npu.utils.utils import _is_interactive_command_line


class _SubModuleID(Enum):
    PTA = 0
    OPS = 1
    DIST = 2
    GRAPH = 3
    PROF = 4
    UNKNOWN = 99


class ErrCode(Enum):
    SUCCESS = (0, "success")
    PARAM = (1, "invalid parameter")
    TYPE = (2, "invalid type")
    VALUE = (3, "invalid value")
    PTR = (4, "invalid pointer")
    INTERNAL = (5, "internal error")
    MEMORY = (6, "memory error")
    NOT_SUPPORT = (7, "feature not supported")
    NOT_FOUND = (8, "resource not found")
    UNAVAIL = (9, "resource unavailable")
    SYSCALL = (10, "system call failed")
    TIMEOUT = (11, "timeout error")
    PERMISSION = (12, "permission error")
    ACL = (100, "call acl api failed")
    HCCL = (200, "call hccl api failed")
    GE = (300, "call ge api failed")
    EXCEPT = (999, "applicaiton exception")

    @property
    def code(self):
        return self.value[0]

    @property
    def msg(self):
        return self.value[1]


def _format_error_msg(submodule, error_code):
    def get_device_id():
        try:
            import torch_npu._C
            return torch_npu._C._npu_getLocalDevice()
        except Exception:
            return -1

    def get_rank_id():
        try:
            import torch.distributed as dist
            rank = dist.get_rank()
            return rank
        except Exception:
            return -1

    error_msg = ""
    if not get_env_compact_error_output() and not _is_interactive_command_line():
        error_msg += "\n[ERROR] {time} (PID:{pid}, Device:{device}, RankID:{rank}) {error_code} {submodule_name} {error_code_msg}"

    return error_msg.format(
        time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
        pid=os.getpid(),
        device=get_device_id(),
        rank=get_rank_id(),
        error_code="ERR{:0>2d}{:0>3d}".format(submodule.value, error_code.code, ),
        submodule_name=submodule.name,
        error_code_msg=error_code.msg)


def pta_error(error: ErrCode) -> str:
    return _format_error_msg(_SubModuleID.PTA, error)


def ops_error(error: ErrCode) -> str:
    return _format_error_msg(_SubModuleID.OPS, error)


def dist_error(error: ErrCode) -> str:
    return _format_error_msg(_SubModuleID.DIST, error)


def graph_error(error: ErrCode) -> str:
    return _format_error_msg(_SubModuleID.GRAPH, error)


def prof_error(error: ErrCode) -> str:
    return _format_error_msg(_SubModuleID.PROF, error)


def get_env_compact_error_output():
    return int(os.getenv("TORCH_NPU_COMPACT_ERROR_OUTPUT", "0"))


class _NPUExceptionHandler(object):
    def __init__(self):
        self.exception = None
        self.npu_exception = "\[ERROR\] [0-9\-\:]* \(PID:\d*, Device:\-?\d*, RankID:\-?\d*\) ERR\d{5}"
        self.npu_timeout_exception = "error code is 107020"
        self.npu_timeout_exit_offset = 3
        self.force_stop_flag = False

    def _is_exception(self, exception_pattern):
        if self.exception and re.search(exception_pattern, self.exception):
            return True
        return False

    def set_force_stop_exception(self, flag):
        self.force_stop_flag = flag

    def _excepthook(self, exc_type, exc, *args):
        if self.force_stop_flag:
            exc_type = RuntimeError
            exc = RuntimeError("FORCE STOP." + pta_error(ErrCode.ACL))
        self.exception = str(exc)
        self._origin_excepthook(exc_type, exc, *args)

    def patch_excepthook(self):
        self._origin_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

    def handle_exception(self):
        # exception raised by other component, such as original PyTorch, third-party library, or application code.
        if self.exception:
            if self.force_stop_flag:
                raise RuntimeError("FORCE STOP." + pta_error(ErrCode.ACL))
            if self._is_exception(self.npu_exception):
                if self._is_exception(self.npu_timeout_exception) or get_env_compact_error_output():
                    # if npu timeout, let other processes exit properly before elastic agent kills them.
                    time.sleep(self.npu_timeout_exit_offset)
            else:
                print(_format_error_msg(_SubModuleID.UNKNOWN, ErrCode.EXCEPT).lstrip())


_except_handler = _NPUExceptionHandler()
