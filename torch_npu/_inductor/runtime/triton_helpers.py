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