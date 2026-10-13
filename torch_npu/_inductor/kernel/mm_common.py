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
