import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

from .config import logging

try:
    extension = tl.extra.cann.extension
    libdevice = tl.extra.cann.libdevice
except Exception as e:
    logging.debug(f"import tl.extra.cann.extension or tl.extra.cann.libdevice error: {e}")
    libdevice = tl.extra.ascend.libdevice

math = tl.math


@triton.jit
def maximum(a, b):
    return tl.maximum(a, b, tl.PropagateNan.ALL)


@triton.jit
def minimum(a, b):
    return tl.minimum(a, b, tl.PropagateNan.ALL)


triton_helpers.maximum = maximum
triton_helpers.minimum = minimum


@triton.jit
def frexp(x):
    y = libdevice.ilogb(x) + 1
    exponent = tl.where(x == 0, 0, y)
    mantissa = tl.where(x == 0, 0, libdevice.ldexp(x, -y))
    return mantissa, exponent