import logging
import triton
import triton.language as tl

from torch._inductor.runtime.triton_helpers import *  # noqa: F401, F403

try:
    extension = tl.extra.cann.extension
    libdevice = tl.extra.cann.libdevice
except Exception as e:
    logging.debug("import tl.extra.cann.extension or tl.extra.cann.libdevice error: %s", e)  # noqa: G200
    libdevice = tl.extra.ascend.libdevice

math = tl.math

@triton.jit
def frexp(x):
    y = libdevice.ilogb(x) + 1
    exponent = tl.where(x == 0, 0, y)
    mantissa = tl.where(x == 0, 0, libdevice.ldexp(x, -y))
    return mantissa, exponent
