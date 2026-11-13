import logging
import triton
import triton.language as tl

from torch._inductor.runtime.triton_helpers import *  # noqa: F401, F403
from torch._inductor.runtime.triton_helpers import (
    maximum,
    minimum,
    promote_to_tensor,
    select_one,
)

try:
    extension = tl.extra.cann.extension
    libdevice = tl.extra.cann.libdevice
except Exception as e:
    logging.debug("import tl.extra.cann.extension or tl.extra.cann.libdevice error: %s", e)  # noqa: G200
    libdevice = tl.extra.ascend.libdevice

math = tl.math

@triton.jit
def frexp(x):
    # dtype-agnostic entry: fp16/bf16 -> fp32 is exact (value-preserving),
    # and fp16/bf16 subnormals become *normal* fp32 -> no subnormal path needed for them.
    xf = x.to(tl.float32)
    xi = xf.to(tl.uint32, bitcast=True)
    exponent_bits = (xi >> 23) & 0xFF
    # int32 before debias: uint32 arithmetic wraps for |x| < 1 (negative exponents)
    exponent = exponent_bits.to(tl.int32) - 126

    mantissa_bits = (xi & 0x807FFFFF) | 0x3F000000
    mantissa = mantissa_bits.to(tl.float32, bitcast=True)
    is_zero = xf == 0.0

    exponent = tl.where(is_zero, 0, exponent)
    mantissa = tl.where(is_zero, xf, mantissa)   # xf, not 0.0: preserves -0.0 sign like torch.frexp
    # inf/nan (exp field all ones): mantissa passes through, exponent 0
    is_special = exponent_bits == 0xFF
    exponent = tl.where(is_special, 0, exponent)
    mantissa = tl.where(is_special, xf, mantissa)
    # same significand as x -> exact downcast
    return mantissa.to(x.dtype), exponent


@triton.jit
def _restore_reduced_dim(reduced, dim: tl.constexpr, ndim: tl.constexpr):
    """Put back the axis a reduce removed, so the result broadcasts again."""
    if ndim == 1:
        return promote_to_tensor(reduced)
    return tl.expand_dims(reduced, dim)


@triton.jit
def _minimum_propagate_nan(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _extremum(value, dim: tl.constexpr, want_max: tl.constexpr):
    """Keep integer extrema on comparison-based reductions.

    Native integer min/max may return a value absent from the input lanes,
    leaving the following index reduction with no match. Floating extrema
    propagate NaNs so the index reduction can select the first NaN lane.
    """
    if value.dtype.is_floating():
        # Match tl.max's promotion and keep the min combine's input/output
        # types consistent for fp16 and bf16.
        if tl.constexpr(value.dtype.primitive_bitwidth) < tl.constexpr(32):
            value = value.to(tl.float32)
        if want_max:
            return tl.max(value, dim, propagate_nan=True)
        else:
            return tl.reduce(value, dim, _minimum_propagate_nan)
    else:
        if want_max:
            return tl.reduce(value, dim, maximum)
        else:
            return tl.reduce(value, dim, minimum)


@triton.jit
def max_with_index(value, index, dim: tl.constexpr):
    """Override upstream max_with_index to avoid a two-source tl.reduce.

    Upstream reduces (value, index) together with tl.reduce. Fed the loop-carried
    accumulators of a looped reduction, NPU's backend returns the lane coordinate
    instead of the carried index, so an argmax over a dynamic reduction axis comes
    out as the position within one tile.

    Reduce the extremum, then the lowest carried index among matching lanes.
    For floats, return that lane's original value, preserving NaN payloads
    and signed zero like upstream's paired reduction.
    """
    peak = _extremum(value, dim, True)
    peak_bcast = _restore_reduced_dim(peak, dim, len(value.shape))
    filler = _restore_reduced_dim(tl.max(index, dim), dim, len(value.shape))
    matches = value == peak_bcast
    if value.dtype.is_floating():
        matches = matches | ((value != value) & (peak_bcast != peak_bcast))
    selected_index = tl.min(tl.where(matches, index, filler), dim)
    if value.dtype.is_floating():
        selected = index == _restore_reduced_dim(selected_index, dim, len(value.shape))
        peak = select_one(value, selected, dim)
    return peak, selected_index


@triton.jit
def min_with_index(value, index, dim: tl.constexpr):
    """Mirror of max_with_index; see there for why the two-source reduce is out."""
    valley = _extremum(value, dim, False)
    valley_bcast = _restore_reduced_dim(valley, dim, len(value.shape))
    filler = _restore_reduced_dim(tl.max(index, dim), dim, len(value.shape))
    matches = value == valley_bcast
    if value.dtype.is_floating():
        matches = matches | ((value != value) & (valley_bcast != valley_bcast))
    selected_index = tl.min(tl.where(matches, index, filler), dim)
    if value.dtype.is_floating():
        selected = index == _restore_reduced_dim(selected_index, dim, len(value.shape))
        valley = select_one(value, selected, dim)
    return valley, selected_index


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
