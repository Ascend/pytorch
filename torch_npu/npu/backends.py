"""
This file will be discarded.
"""

from torch_npu.npu._backends import flash_sdp_enabled, enable_flash_sdp, mem_efficient_sdp_enabled, \
    enable_mem_efficient_sdp, math_sdp_enabled, enable_math_sdp, sdp_kernel, preferred_linalg_library

from torch_npu.npu._fft_plan_cache import NPUFFTPlanCache

__all__ = []

fft_plan_cache = NPUFFTPlanCache()
