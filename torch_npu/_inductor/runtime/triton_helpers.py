import logging
import triton
import triton.language as tl

from torch._inductor.runtime.triton_helpers import *

try:
    extension = tl.extra.cann.extension
    libdevice = tl.extra.cann.libdevice
except Exception as e:
    logging.debug(f"import tl.extra.cann.extension or tl.extra.cann.libdevice error: {e}")
    libdevice = tl.extra.ascend.libdevice

math = tl.math

@triton.jit
def frexp(x):
    y = libdevice.ilogb(x) + 1
    exponent = tl.where(x == 0, 0, y)
    mantissa = tl.where(x == 0, 0, libdevice.ldexp(x, -y))
    return mantissa, exponent


@triton.jit
def _welford_sum_combine(a, b):
    return a + b


@triton.jit
def welford(mean, m2, weight, dim):
    """Override upstream welford to avoid tl.reduce with 3 source operands.

    NPU's Triton compiler (TritonOpConverter.cpp) asserts srcs.size() <= 2,
    but upstream welford uses tl.reduce((mean,m2,weight), dim, welford_combine)
    with 3 operands. This version decomposes into 3 single-operand reduces.
    """
    sum_x = weight * mean
    sum_x2 = m2 + weight * mean * mean
    total_sum_x = tl.reduce(sum_x, dim, _welford_sum_combine)
    total_sum_x2 = tl.reduce(sum_x2, dim, _welford_sum_combine)
    total_weight = tl.reduce(weight, dim, _welford_sum_combine)
    new_mean = tl.where(total_weight == 0.0, 0.0, total_sum_x / total_weight)
    new_m2 = total_sum_x2 - total_weight * new_mean * new_mean
    return new_mean, new_m2, total_weight
