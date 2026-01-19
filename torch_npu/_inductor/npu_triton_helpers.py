import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

libdevice = tl.extra.cann.libdevice
extension = tl.extra.cann.extension
math = tl.math


@triton.jit
def maximum(a, b):
    return tl.maximum(a, b, tl.PropagateNan.ALL)


@triton.jit
def minimum(a, b):
    return tl.minimum(a, b, tl.PropagateNan.ALL)


triton_helpers.maximum = maximum
triton_helpers.minimum = minimum
