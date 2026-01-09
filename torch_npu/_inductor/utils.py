from __future__ import annotations

import functools
from typing import List, Optional

import torch
import torch_npu


# Not good implementation, but no other way
def get_current_raw_stream(device):
    return torch.npu.current_stream(device).npu_stream


def patch_is_same_tensor():
    from torch._subclasses.fake_tensor import FakeTensor

    def is_same_tensor(data: torch.Tensor, value: torch.Tensor):
        if isinstance(data, FakeTensor) or isinstance(value, FakeTensor):
            return False
        return (
            not data.is_mkldnn
            and data.size() == value.size()
            and data.stride() == value.stride()
            and data.dtype == value.dtype
            and data.device == value.device
            and data.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            and data.storage_offset() == value.storage_offset()
        )
    
    from torch._inductor import utils, graph
    utils.is_same_tensor = is_same_tensor
    # We need to do extra-patch because of code like `from xxx import is_same_tensor`
    graph.is_same_tensor = is_same_tensor


def patch_is_gpu():
    from torch._inductor.utils import GPU_TYPES
    GPU_TYPES.append('npu')


def patch_has_triton():
    from torch.utils._triton import has_triton_package

    @functools.lru_cache(None)
    def has_triton() -> bool:
        if not has_triton_package():
            return False

        from torch._dynamo.device_interface import get_interface_for_device

        def cuda_extra_check(device_interface):
            return True

        def cpu_extra_check(device_interface):
            import triton.backends

            return "cpu" in triton.backends.backends

        def _return_true(device_interface):
            return True

        triton_supported_devices = {
            "cuda": cuda_extra_check,
            "xpu": _return_true,
            "cpu": cpu_extra_check,
            "npu": _return_true
        }

        def is_device_compatible_with_triton():
            for device, extra_check in triton_supported_devices.items():
                device_interface = get_interface_for_device(device)
                if device_interface.is_available() and extra_check(device_interface):
                    return True
            return False

        return is_device_compatible_with_triton()

    torch.utils._triton.has_triton = has_triton
    torch._inductor.scheduler.has_triton = has_triton


def disable_foreach():
    from torch._inductor.scheduler import Scheduler

    def create_foreach_nodes(self):
        return

    Scheduler.create_foreach_nodes = create_foreach_nodes


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


def _use_template_for_npu(layout, allowed_layout_dtypes: List[torch.dtype]) -> bool:
    return layout.device.type == "npu" and layout.dtype in allowed_layout_dtypes


def use_triton_template(
    layout: Layout, *, enable_int32: bool = False, enable_float8: bool = False
) -> bool:
    from torch._inductor.utils import is_gpu, _use_autotune_backend, use_max_autotune
    from torch._inductor.codegen.common import BackendFeature, has_backend_feature

    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if enable_int32:
        layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    if enable_float8:
        layout_dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
    return (
        (
            (
                is_gpu(layout.device.type)
                and _use_template_for_npu(layout, layout_dtypes)
            )
            or (layout.device.type == "cpu" and layout.dtype in layout_dtypes)
        )
        and use_max_autotune()
        and _use_autotune_backend("TRITON")
        and has_backend_feature(layout.device, BackendFeature.TRITON_TEMPLATES)
    )


def use_catlass_template(op_name: str, layout: Layout, m: int, n: int, k: int) -> bool:
    from torch._inductor.virtualized import V
    from torch._inductor.utils import _use_autotune_backend, use_max_autotune

    from .config import catlass as catlass_config
    enabled_ops = catlass_config.catlass_enabled_ops.upper()
    if enabled_ops == "ALL":
        pass
    elif not op_name.upper() in [x.strip() for x in enabled_ops.split(",")]:
        return False

    gemm_size = V.graph.sizevars.size_hint(m * n * k, fallback=-1)
    if gemm_size <= 0 or gemm_size < catlass_config.catlass_backend_min_gemm_size:
        return False

    # Do not use catlass template on ROCm
    if torch.version.hip:
        return False

    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    res = (
        _use_template_for_npu(layout, layout_dtypes)
        and use_max_autotune()
        and _use_autotune_backend("CATLASS")
    )

    return res