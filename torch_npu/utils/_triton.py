import functools
import torch
from torch.utils._triton import has_triton_package
import torch_npu


def has_triton() -> bool:
    # here has_triton only return False,
    # when has_triton() is True, config.triton.autotune_at_compile_time is True,
    # AOTI is not currently supported for autotune at compile stage
    return False


def has_triton_tma():
    # here has_triton_tma only return False,
    # keep pace with no transfer_to_npu, will be fully implemented in future
    return False


def has_triton_tma_device():
    # here has_triton_tma_device only return False,
    # keep pace with no transfer_to_npu, will be fully implemented in future
    return False


def patch_triton_for_origin_func():
    patch_has_triton(torch.utils._triton)
    patch_has_triton_tma(torch.utils._triton)
    patch_has_triton_tma_device(torch.utils._triton)


def patch_triton_for_dynamo():
    patch_has_triton(torch._dynamo.utils)


def patch_triton_for_inductor():
    patch_has_triton(torch._inductor.compile_fx)
    patch_has_triton(torch._inductor.scheduler)
    patch_has_triton(torch._inductor.runtime.autotune_cache)
    from torch._inductor.fx_passes import pad_mm
    patch_has_triton(pad_mm)

    patch_has_triton_tma_device(torch._inductor.kernel.mm_scaled)


def patch_has_triton(modules):
    setattr(modules, "has_triton", has_triton)


def patch_has_triton_tma(modules):
    setattr(modules, "has_triton_tma", has_triton_tma)


def patch_has_triton_tma_device(modules):
    setattr(modules, "has_triton_tma_device", has_triton_tma_device)


patch_triton_for_origin_func()