import functools
import torch
from torch.utils._triton import has_triton_package
import torch_npu


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


@functools.lru_cache(None)
def has_triton_tma():
    if has_triton_package():
        if (
            torch_npu.npu.is_available()
            and not torch.version.hip
        ):
            try:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.lru_cache(None)
def has_triton_tma_device():
    if has_triton_package():
        if (
            torch_npu.npu.is_available()
            and not torch.version.hip
        ):
            try:
                from triton.language.extra.ascend.libdevice import (  # noqa: F401
                    reciprocal,
                    log1p,
                )

                return True
            except ImportError:
                pass

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