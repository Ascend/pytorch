import torch

from torch_npu._compat._impl import compat_impl, compat_impl_container

__all__ = ["get_default_generator"]


# Upstream 2.14 added torch._C._accelerator_getDefaultGenerator, the unified
# entry point for a backend's default generator. 2.13 and earlier have no
# accelerator-level equivalent, so fall back to the NPU-specific
# default_generators tuple.
@compat_impl(key="get_default_generator", ge=(2, 14))
def get_default_generator_upstream(device_index: int):
    return torch._C._accelerator_getDefaultGenerator(device_index)


@compat_impl(key="get_default_generator", lt=(2, 14))
def get_default_generator_local(device_index: int):
    return torch.npu.default_generators[device_index]


get_default_generator = compat_impl_container["get_default_generator"].resolve()
