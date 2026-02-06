import torch
import torch_npu

__all__ = ["get_amp_supported_dtype", "is_autocast_enabled", "set_autocast_enabled", "get_autocast_dtype",
           "set_autocast_dtype"]


def get_amp_supported_dtype():
    if torch.npu.is_bf16_supported():
        return [torch.float16, torch.bfloat16, torch.float32]
    return [torch.float16, torch.float32] 


def is_autocast_enabled():
    return torch_npu._C.is_autocast_enabled()


def set_autocast_enabled(enable):
    torch_npu._C.set_autocast_enabled(enable)


def get_autocast_dtype():
    return torch_npu._C.get_autocast_dtype()


def set_autocast_dtype(dtype):
    return torch_npu._C.set_autocast_dtype(dtype)
