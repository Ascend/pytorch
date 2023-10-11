import contextlib
from typing import Union
import warnings

import torch
import torch_npu._C


def flash_sdp_enabled() -> bool:
    r"""
    .. warning:: This flag is beta and subject to change.
    Returns whether flash scaled dot product attention is enabled or not.
    """
    warnings.warn("Currently, the device operator does not support flash sdp and only sets Global variable!")
    return torch._C._get_flash_sdp_enabled()


def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.
    Enables or disables flash scaled dot product attention.
    """
    warnings.warn("Currently, the device operator does not support flash sdp and only sets Global variable!")
    torch._C._set_sdp_use_flash(enabled)


def mem_efficient_sdp_enabled() -> bool:
    r"""
    .. warning:: This flag is beta and subject to change.
    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
    warnings.warn("Currently, the device operator does not support mem_efficient sdp and only sets Global variable!")
    return torch._C._get_mem_efficient_sdp_enabled()


def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.
    Enables or disables memory efficient scaled dot product attention.
    """
    warnings.warn("Currently, the device operator does not support mem_efficient sdp and only sets Global variable!")
    torch._C._set_sdp_use_mem_efficient(enabled)


def math_sdp_enabled() -> bool:
    r"""
    .. warning:: This flag is beta and subject to change.
    Returns whether math scaled dot product attention is enabled or not.
    """
    warnings.warn("Currently, the device operator does not support math sdp and only sets Global variable!")
    return torch._C._get_math_sdp_enabled()


def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.
    Enables or disables math scaled dot product attention.
    """
    warnings.warn("Currently, the device operator does not support math sdp and only sets Global variable!")
    torch._C._set_sdp_use_math(enabled)


@contextlib.contextmanager
def sdp_kernel(enable_flash: bool = True, enable_math: bool = True, enable_mem_efficient: bool = True):
    r"""
    .. warning:: This flag is beta and subject to change.
    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product
    attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    warnings.warn("Currently, the device operator does not support flash、math、mem_efficient sdp "
                  "and only sets Global variable!")
    previous_flash: bool = flash_sdp_enabled()
    previous_mem_efficient: bool = mem_efficient_sdp_enabled()
    previous_math: bool = math_sdp_enabled()
    try:
        enable_flash_sdp(enable_flash)
        enable_mem_efficient_sdp(enable_mem_efficient)
        enable_math_sdp(enable_math)
        yield {}
    finally:
        enable_flash_sdp(previous_flash)
        enable_mem_efficient_sdp(previous_mem_efficient)
        enable_math_sdp(previous_math)


def preferred_linalg_library(backend: Union[None, str, torch._C._LinalgBackend] = None) -> torch._C._LinalgBackend:
    """Currently, the linalg lib is not available and does not support it.
    By default, it returns Default type.
    """
    warnings.warn("torch.npu.preferred_linalg_library isn't implemented!")
    return torch._C._LinalgBackend.Default
