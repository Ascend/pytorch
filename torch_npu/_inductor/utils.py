from __future__ import annotations

import functools
import logging

import torch

import torch._inductor.config as inductor_config
log = logging.getLogger("torch._inductor")


def get_current_raw_stream(device):
    return torch.npu.current_stream(device).npu_stream


def patch_is_gpu():
    from torch._inductor.utils import GPU_TYPES, get_gpu_type

    if "npu" not in GPU_TYPES:
        GPU_TYPES.append("npu")
    get_gpu_type.cache_clear()


def resolve_npu_device_index(device_idx=None) -> int:
    from torch._inductor.utils import decode_device

    return decode_device(torch.device("npu", device_idx)).index


def patch_has_triton():
    from torch._inductor import compile_fx
    from torch_npu.utils._dynamo import has_triton

    torch._inductor.scheduler.has_triton = has_triton
    compile_fx.has_triton = has_triton


def patch_device_supports_tma():
    @functools.lru_cache(None)
    def _device_supports_tma():
        return torch.npu.is_available() and not torch.version.hip

    torch.utils._triton._device_supports_tma = _device_supports_tma


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


def _use_template_for_npu(layout, allowed_layout_dtypes: list[torch.dtype]) -> bool:
    return layout.device.type == "npu" and layout.dtype in allowed_layout_dtypes


def use_triton_template(
    layout, *, enable_int32: bool = False, enable_float8: bool = False
) -> bool:
    from torch._inductor.codegen.common import BackendFeature, has_backend_feature
    from torch._inductor.utils import _use_autotune_backend, is_gpu

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
        and inductor_config.max_autotune
        and _use_autotune_backend("TRITON")
        and has_backend_feature(layout.device, BackendFeature.TRITON_TEMPLATES)
    )


def use_catlass_template(op_name, layout, m: int, n: int, k: int) -> bool:
    from torch._inductor.utils import _use_autotune_backend
    from torch._inductor.virtualized import V

    from .codegen.catlass.catlass_utils import try_import_catlass
    from .config import catlass as catlass_config

    enabled_ops = catlass_config.catlass_enabled_ops.upper()
    if enabled_ops == "ALL":
        pass
    elif op_name.upper() not in [x.strip() for x in enabled_ops.split(",")]:
        return False

    gemm_size = V.graph.sizevars.optimization_hint(m * n * k, fallback=-1)
    if gemm_size <= 0 or gemm_size < catlass_config.catlass_backend_min_gemm_size:
        return False

    # Do not use catlass template on ROCm
    if torch.version.hip:
        return False

    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    res = (
        _use_template_for_npu(layout, layout_dtypes)
        and inductor_config.max_autotune
        and _use_autotune_backend("CATLASS")
    )

    if res:
        if not try_import_catlass():
            log.warning(
                "Failed to import CATLASS lib. Please check whether "
                "_inductor.config.catlass.catlass_dir is set correctly. "
                "Skipping CATLASS backend for now"
            )
            return False

    return res

def triton_support_auto_blockify():
    from triton.backends.ascend.utils import _is_auto_map_parallel_blocks_enabled
    return _is_auto_map_parallel_blocks_enabled()

def triton_support_ffts():
    from triton.backends.ascend.utils import (
        force_disable_ffts,
        get_ascend_arch_from_env,
        is_ffts_supported,
    )

    arch = get_ascend_arch_from_env()
    return is_ffts_supported(arch) and (not force_disable_ffts())
