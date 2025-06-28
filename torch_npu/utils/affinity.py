__all__ = []

from typing import List

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


def _set_thread_affinity(core_range: List[int] = None):
    if core_range is None:
        torch_npu._C._npu_set_thread_affinity(-1, -1)
    elif (len(core_range) == 2):
        if core_range[0] < 0 or core_range[1] < 0:
            raise ValueError("Core range should be nonnegative." + pta_error(ErrCode.PARAM))
        torch_npu._C._npu_set_thread_affinity(core_range[0], core_range[1])
    else:
        raise ValueError("The length of input list of set_thread_affinity should be 2." + pta_error(ErrCode.PARAM))


def _reset_thread_affinity():
    torch_npu._C._npu_reset_thread_affinity()