import torch

from ..config import num_cube_core


# Cube-side on-chip capacities for Atlas A5, from catlass/arch/arch.hpp.  The
# device properties do not report them, so they stay literals here.
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
L0C_BYTES = 256 * 1024
L1_BYTES = 512 * 1024

# One MMAD fractal is 16x16 elements.  A panel narrower than a fractal still
# costs a whole one, so a tile extent is budgeted on its rounded-up size rather
# than on the raw one.
MMAD_M_FRACTAL = 16
MMAD_K_FRACTAL = 16

# The accumulator is fp32 whatever the operands are, so an L0C tile is always
# budgeted at four bytes per element while the operand tiles scale with dtype.
ACCUM_BYTES = 4

# Half of L0C is the budget a tile is sized against: the other half is the
# second accumulator buffer, which is what lets the next tile's mmad run under
# this one's fixpipe.  A tile may claim the whole of L0C instead by asking the
# backend not to buffer it, at the cost of that overlap.
L0C_DOUBLE_BUFFERED_BYTES = L0C_BYTES // 2

# The compile option naming how a kernel wants L0C buffered, and its two
# values.  It has to be a compile option and not a tl.constexpr: a key the
# template does not declare never reaches bishengir-compile.
L0C_BUFFERING_KEY = "limit_auto_multi_buffer_of_local_buffer"
L0C_DOUBLE_BUFFERED = "no-limit"
L0C_SINGLE_BUFFERED = "no-l0c"

# Operand element size assumed when the dtype is unknown.  Two bytes is the
# conservative choice: it is what the capacity checks are tuned at, and
# guessing smaller would admit plans that overrun L1 on the device.
DEFAULT_ELEM_BYTES = 2

# Resolved the same way the rest of the NPU inductor backend resolves it, so a
# non-28-core part and an NPU_DEVICE_LIMIT override are both honoured.
NUM_CUBE_CORES = num_cube_core

# How many block-loop iterations each core should get, so that consecutive
# iterations have something to overlap with.
MIN_WAVES = 3

# Copies of the A/B L1 staging buffers in the batched kernel's group loop.
#
# Without them the group loop cannot overlap at all: iteration i+1's nd2nz
# writes the same L1 buffer iteration i's mmad is still reading through MTE1,
# so MTE2 stalls on the WAR hazard.  MarkMultiBuffer would mark these
# automatically, but only if it can trace the alloc to an enclosing scf.for, so
# it is worth stating outright rather than relying on the loop surviving
# canonicalisation.
L1_BUFFER_COPIES = 2


_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
}


def dtype_to_bytes(dtype):
    return _DTYPE_BYTES.get(dtype, 0)


def round_up_to_16(x):
    return ((x + 15) // 16) * 16


def round_down_to_16(x):
    return (x // 16) * 16
