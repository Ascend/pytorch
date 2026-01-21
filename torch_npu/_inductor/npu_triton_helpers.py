import triton
import triton.language as tl

try:
    import triton.language.extra.cann.libdevice as libdevice
except ImportError:
    import triton.language.extra.ascend.libdevice as libdevice
from torch._inductor.runtime import triton_helpers

try:
    libdevice = tl.extra.cann.libdevice
except AttributeError:
    libdevice = tl.extra.ascend.libdevice
math = tl.math


@triton.jit
def maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def minimum(a, b):
    return tl.minimum(a, b)


triton_helpers.maximum = maximum
triton_helpers.minimum = minimum
