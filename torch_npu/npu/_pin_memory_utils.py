import torch_npu

from torch_npu.utils._error_code import ErrCode, pta_error

__all__ = ["pin_memory", "unpin_memory"]


# CANN flag for registering existing host memory as page-locked memory.
_ACL_HOST_REG_PINNED = 0x10000000


def pin_memory(data_ptr: int, size: int) -> None:
    result = torch_npu.npu.npurt().npuHostRegister(
        data_ptr,
        size,
        _ACL_HOST_REG_PINNED,
    )
    if result != 0:
        raise RuntimeError(
            f"npuHostRegister failed with error code {result}."
            + pta_error(ErrCode.ACL)
        )


def unpin_memory(data_ptr: int) -> None:
    result = torch_npu.npu.npurt().npuHostUnregister(data_ptr)
    if result != 0:
        raise RuntimeError(
            f"npuHostUnregister failed with error code {result}."
            + pta_error(ErrCode.ACL)
        )
