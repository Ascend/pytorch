import os
import time
from enum import Enum


class SubModuleID(Enum):
    PTA = 0
    OPS = 1
    DIST = 2
    GRAPH = 3
    PROF = 4


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

    @property
    def code(self):
        return self.value[0]

    @property
    def msg(self):
        return self.value[1]


def format_error_msg(submodule, error_code):
    def get_device_id():
        try:
            from torch_npu.npu import current_device
            device = current_device()
            return device
        except Exception:
            return -1

    def get_rank_id():
        try:
            import torch.distributed as dist
            rank = dist.get_rank()
            return rank
        except Exception:
            return -1

    error_msg = "\n[ERROR] {time} (PID:{pid}, Device:{device}, RankID:{rank}) {error_code} {submodule_name} {error_code_msg}"

    return error_msg.format(
            time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
            pid=os.getpid(),
            device=get_device_id(),
            rank=get_rank_id(),
            error_code="ERR{:0>2d}{:0>3d}".format(submodule.value, error_code.code,),
            submodule_name=submodule.name,
            error_code_msg=error_code.msg)


def pta_error(error: ErrCode) -> str:
    return format_error_msg(SubModuleID.PTA, error)


def ops_error(error: ErrCode) -> str:
    return format_error_msg(SubModuleID.OPS, error)


def dist_error(error: ErrCode) -> str:
    return format_error_msg(SubModuleID.DIST, error)


def graph_error(error: ErrCode) -> str:
    return format_error_msg(SubModuleID.GRAPH, error)


def prof_error(error: ErrCode) -> str:
    return format_error_msg(SubModuleID.PROF, error)
