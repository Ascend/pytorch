import logging
import triton
import triton.language as tl

from torch._inductor.runtime.triton_helpers import *
from torch._inductor.runtime.triton_helpers import promote_to_tensor

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
def _restore_reduced_dim(reduced, dim: tl.constexpr, ndim: tl.constexpr):
    """Put back the axis a reduce removed, so the result broadcasts again."""
    if ndim == 1:
        return promote_to_tensor(reduced)
    return tl.expand_dims(reduced, dim)


@triton.jit
def max_with_index(value, index, dim: tl.constexpr):
    """Override upstream max_with_index to avoid a two-source tl.reduce.

    Upstream reduces (value, index) together with tl.reduce. Fed the loop-carried
    accumulators of a looped reduction, NPU's backend returns the lane coordinate
    instead of the carried index, so an argmax over a dynamic reduction axis comes
    out as the position within one tile. Two single-operand reduces give the same
    answer: the extremum, then the lowest index among the elements reaching it.
    The largest index of the slice is the filler for elements that are not at the
    extremum, which avoids a dtype-dependent sentinel and never wins the min.

    Both reduces are builtins. A tl.reduce carrying a combine region compiles
    down a slower path on this backend, and propagate_nan gives the same NaN
    ordering that upstream's maximum would.
    """
    peak = tl.max(value, dim, propagate_nan=True)
    peak_bcast = _restore_reduced_dim(peak, dim, len(value.shape))
    filler = _restore_reduced_dim(tl.max(index, dim), dim, len(value.shape))
    # A NaN outranks every number and is not equal to itself, so match it apart.
    at_peak = (value == peak_bcast) | ((value != value) & (peak_bcast != peak_bcast))
    return peak, tl.min(tl.where(at_peak, index, filler), dim)


@triton.jit
def min_with_index(value, index, dim: tl.constexpr):
    """Mirror of max_with_index; see there for why the two-source reduce is out."""
    valley = tl.min(value, dim, propagate_nan=True)
    valley_bcast = _restore_reduced_dim(valley, dim, len(value.shape))
    filler = _restore_reduced_dim(tl.max(index, dim), dim, len(value.shape))
    at_valley = (value == valley_bcast) | (
        (value != value) & (valley_bcast != valley_bcast)
    )
    return valley, tl.min(tl.where(at_valley, index, filler), dim)


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
