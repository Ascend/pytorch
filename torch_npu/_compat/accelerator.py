import torch

from torch_npu._compat.version import CURRENT_VERSION

__all__ = ["get_default_generator"]


# COMPAT(>= 2.14): upstream added
#   torch._C._accelerator_getDefaultGenerator, the unified entry point for a
#   backend's default generator. 2.13 and earlier have no accelerator-level
#   equivalent, so fall back to the NPU-specific default_generators tuple.
def get_default_generator(device_index: int):
    if CURRENT_VERSION >= (2, 14):
        return torch._C._accelerator_getDefaultGenerator(device_index)
    return torch.npu.default_generators[device_index]
