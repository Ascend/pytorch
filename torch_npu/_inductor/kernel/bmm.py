import functools
import itertools
import logging
from typing import Any, Dict, List, Tuple

import torch
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate

from torch._inductor import ir, lowering as L
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
)
from torch._inductor.utils import (
    use_aten_gemm_kernels,
    use_ck_template,
    use_cpp_bmm_template,
    sympy_product,
)
from torch._inductor.virtualized import V
from torch._inductor.kernel.mm_common import (
    _is_static_problem,
    mm_args,
)
from torch._inductor.kernel import bmm as inductor_bmm

from .mm import is_contiguous_striding
from .optimize_tiling import get_optimized_tiling
from ..select_algorithm import NPUTritonTemplate, NPUTemplateCompileOption
from ..utils import use_catlass_template, use_triton_template
from torch_npu.npu import matmul
from .mm_common import (
    ACCUM_BYTES,
    DEFAULT_ELEM_BYTES,
    dtype_to_bytes,
    L0A_BYTES,
    L0C_BUFFERING_KEY,
    L0C_BYTES,
    L0C_DOUBLE_BUFFERED,
    L0C_DOUBLE_BUFFERED_BYTES,
    L0C_SINGLE_BUFFERED,
    L1_BUFFER_COPIES,
    L1_BYTES,
    MIN_WAVES,
    MMAD_K_FRACTAL,
    MMAD_M_FRACTAL,
    NUM_CUBE_CORES,
)


log = logging.getLogger("torch._inductor")
aten = torch.ops.aten

aten_bmm = inductor_bmm.aten_bmm
aten_baddbmm = inductor_bmm.aten_baddbmm


# ---------------------------------------------------------------------------
# Shared by every template below
# ---------------------------------------------------------------------------

def _elem_bytes(dtype) -> int:
    """One operand element in bytes, never under-reported.

    Every template budgets L0 or L1 against this, so a size smaller than the
    truth admits a plan that overruns the buffer on the device.  dtype_to_bytes
    answers 0 for a dtype it does not list, and fp64 is one of those: a torch
    dtype knows its own itemsize, so an unlisted one is budgeted at that.  The
    assumed size is reached only by something that cannot say, and the tiers
    that care about the dtype gate on the dtype itself rather than on this.
    """
    return (dtype_to_bytes(dtype)
            or getattr(dtype, "itemsize", 0)
            or DEFAULT_ELEM_BYTES)


def _l0c_buffering(accum_bytes: int) -> str:
    """How this tile wants L0C buffered, from the accumulator it claims.

    A tile inside half of L0C leaves room for the second buffer and takes it,
    since giving that buffer up costs the overlap between one tile's fixpipe
    and the next tile's mmad.  A wider tile has no choice, single-buffered
    being the only way it compiles at all.
    """
    return (L0C_DOUBLE_BUFFERED if accum_bytes <= L0C_DOUBLE_BUFFERED_BYTES
            else L0C_SINGLE_BUFFERED)


def _split_groups(n_groups: int) -> tuple[int, int, bool]:
    """Split n_groups over programs, as (programs, per_program, ragged).

    A program has to own a *run* of groups: separate program instances cannot
    overlap, each re-paying kernel entry with its pipes drained, while a run
    becomes an scf.for where iteration i+1's MTE2 runs under iteration i's MAC.

    Exact divisors alone leave cores idle whenever the group count has no
    divisor near the core count, which a power-of-two batch never has, so the
    best exact split is weighed against a ragged one.  A group is whole, so
    coming up short costs a trip count and no mask.
    """
    num_cube_cores = NUM_CUBE_CORES
    exact = next((p for p in range(min(n_groups, num_cube_cores), 0, -1)
                  if n_groups % p == 0), 1)
    programs = min(n_groups, num_cube_cores)
    per_program = -(-n_groups // programs)
    if n_groups // exact <= per_program:
        return exact, n_groups // exact, False
    return programs, per_program, True


def _wave_use(tasks: int) -> float:
    """Share of the core-cycles a task count spends on tasks rather than idle."""
    num_cube_cores = NUM_CUBE_CORES
    waves = -(-tasks // num_cube_cores)
    return tasks / (waves * num_cube_cores)


def _window(m_tiles: int) -> int:
    """Height of the T3 window, in M tiles.

    About the square root of the core count, so one wave of consecutive tiles
    covers a block roughly as tall as it is wide.
    """
    return max(1, min(m_tiles, round(NUM_CUBE_CORES ** 0.5)))


class _WideIndexNPUTritonTemplate(NPUTritonTemplate):
    """A template that may index in 64 bits, because it is written to.

    The base class declines any shape whose buffers need 64-bit indexing,
    because a template indexing in int32 wraps silently past 2^31 elements.
    The templates below do not: every offset that scales with the batch is
    taken to INDEX_DTYPE before it is multiplied by a stride, and what remains
    is bounded by M*K, K*N and M*N.

    Without this the large-batch shapes have no candidate at all rather than a
    slow one: every config the generators offer raises here and the shape
    falls back to ATen.
    """

    def _write_index_dtype_define(
        self,
        defines,
        numel,
        buffers,
        index_dtype_override=None,
    ) -> str:
        """Override that falls back to int64 when the base class rejects
        the shape (large buffers needing 64-bit indexing).

        The parent signature now includes an optional ``index_dtype_override``
        parameter.  This override forwards it and returns the resolved dtype
        string, matching the parent's contract.
        """
        try:
            return super()._write_index_dtype_define(
                defines, numel, buffers, index_dtype_override
            )
        except NotImplementedError:
            # The base class raises before writing, so the define is still owed.
            defines.write("INDEX_DTYPE : tl.constexpr = tl.int64\n")
            return "tl.int64"

# Compile options rather than kernel constexprs. A key not listed here is
# emitted as `tl.constexpr` and never reaches bishengir-compile, which silently
# drops --enable-hivm-batch-matmul and lowers the rank-3 dot as a batch loop.
_BMM_COMPILE_OPTIONS = NPUTemplateCompileOption(
    {
        "enable_hivm_batch_matmul": True,
        "unit_flag": False,
        # Only a default: NPUTemplateCompileOption.apply setdefaults, so a
        # config naming this key keeps its own value.  See _l0c_buffering.
        L0C_BUFFERING_KEY: L0C_DOUBLE_BUFFERED,
    }
)


# ---------------------------------------------------------------------------
# The tiling layer: the tiles the generic template and T3 both cut C into
# ---------------------------------------------------------------------------

# The single-buffered tile: the whole of L0C as one square.  Also the extent
# an output needs on both axes to fill it.  See _wants_single_buffered_tile.
WIDE_TILE_N = 256


# Widest derived tile on one axis.  The L0C square is not the bound here: an
# oblong holds the same accumulators, and the L0C test in _derived_tilings is
# what keeps the pair inside L0C.
_MAX_DERIVED_WIDTH = 512

# How many derived tilings join the table.  Enough for the ranking to see a
# choice, few enough that the autotune window still goes mostly to the table.
_MAX_DERIVED_TILINGS = 8


def _low_waste_widths(extent: int, floor: int) -> List[int]:
    """Fractal-aligned widths for one extent, least wasteful first.

    The table of tilings is powers of two, and an extent that is not one pays
    for the rounding in every tile it spans: an extent of 513 under a 64-wide
    tile covers 576, twelve percent in one dimension and compounding across
    three.  An eleven-fractal width covers it in 528 instead, which is why the
    widths worth offering follow from the shape and cannot be tabulated ahead
    of it.

    The floor keeps this from answering with the narrowest width that divides.
    Waste falls monotonically towards a fractal, but below 64 the mmad is too
    small to pay for its own issue and the narrow width loses outright.
    """
    extent = int(extent)
    # An extent under the floor still has one natural width, itself rounded up
    # to a fractal.  Flooring that away returns nothing, and the caller pairs
    # the axes, so one narrow axis would switch the whole derivation off.
    floor = min(floor, -(-extent // MMAD_M_FRACTAL) * MMAD_M_FRACTAL)
    waste_of: Dict[int, int] = {}
    # Counted from the fewest tiles that bring the width under the cap, not
    # from one: K=65536 needs 128 tiles before a width is even admissible.
    first = max(1, -(-extent // _MAX_DERIVED_WIDTH))
    for tiles in range(first, first + 32):
        width = -(-extent // tiles)
        width = -(-width // MMAD_M_FRACTAL) * MMAD_M_FRACTAL
        if width < floor or width > _MAX_DERIVED_WIDTH:
            continue
        # Rounding the width up can empty the last tile, so the count that
        # matters is the one the width actually needs, not the one it came from.
        waste = -(-extent // width) * width - extent
        if waste < waste_of.get(width, waste + 1):
            waste_of[width] = waste
    return sorted(waste_of, key=lambda w: (waste_of[w], -w))


def _k_step(block_m: int, block_n: int, k: int, elem_bytes: int) -> int:
    """How much of K one GM -> L1 staging brings in, for this tile.

    Not a division of K.  Every step accumulates into the same L0C tile and
    the program stores once at the end, so this only sets how often the
    staging buffers turn over; no partial sum leaves the core.

    The answer is the deepest step that keeps A's slab inside L0A and both
    operands double-buffered inside L1.  K is then spread evenly over the
    fewest steps under that depth, so the loop carries no short last turn
    where a full one would have fit.

    L0A is budgeted at its full size and L0B not at all, rather than halving
    both for their double buffers.  The tighter budget was measured and loses
    on most of the shapes it moves: it only shortens the step, and a step
    shorter than the slabs can hold buys no overlap back.
    """
    cap = min(L0A_BYTES // (block_m * elem_bytes),
              L1_BYTES // (L1_BUFFER_COPIES * (block_m + block_n) * elem_bytes),
              -(-k // MMAD_K_FRACTAL) * MMAD_K_FRACTAL)
    cap -= cap % MMAD_K_FRACTAL
    if cap < MMAD_K_FRACTAL:
        return 0
    steps = -(-k // cap)
    step = -(-k // steps)
    step = -(-step // MMAD_K_FRACTAL) * MMAD_K_FRACTAL
    return min(step, cap)


def _k_steps(block_m: int, block_n: int, k: int, elem_bytes: int) -> List[int]:
    """The GM -> L1 staging depths to offer for this tile.

    _k_step gives the deepest one that fits, which is a bound and not an
    optimum.  The ladder below it is powers of two, the form the measured
    optima take, stopping at 256 because a deeper step stages more of K per
    turn but gives MTE2 less to overlap and past 256 the second effect wins.
    Capped at three depths, so the candidate list spends its width on
    geometries rather than on depths.
    """
    cap = _k_step(block_m, block_n, k, elem_bytes)
    if not cap:
        return []
    if k % MMAD_K_FRACTAL == 0 and cap >= k:
        # K is whole fractals and they all stage in one turn, so nothing
        # shallower is worth offering: it only adds turns and leaves a ragged
        # last one.  On the short-K shapes every shallower depth lost.
        return [cap]

    steps = {step for step in (64, 128, 256) if step <= cap} | {cap}
    # A depth past K leaves the kernel no main loop, so the whole reduction
    # runs through the masked tail, which is markedly slower than staging K
    # exactly.  Dropped unless it is all this tile can stage.
    steps = {step for step in steps if step <= k} or {cap}
    return sorted(steps)[-3:]


def _derived_tilings(m: int, n: int, k: int, elem_bytes: int
                     ) -> List[Tuple[int, int]]:
    """(BLOCK_M, BLOCK_N) pairs drawn from the shape, ranked by the mmad volume
    they pad it to.

    Volume is the ranking rather than per-axis waste because the axes trade
    against each other.  A shape the table already fits derives widths that are
    table entries, so it dedupes away and nothing changes.

    The K step is not part of the pair: it follows from it, in _k_step.

    L0A and L0B are not tested.  They are not the wall they look like, a dot
    whose operand slab overruns them being tiled into them by the backend.
    L0C is a real wall -- the accumulator has nowhere to spill and 256x512x128
    faults the core -- and so is L1, which is what the slabs stage through.

    L0C is budgeted whole, so a derived tile may be wide enough to cost the
    second accumulator buffer.  That is left to the config to declare and to
    the bench to judge: the tiles above the double-buffered size are the ones
    that halve a pass over an operand, and on a shape held up by its traffic
    that pays for the lost overlap several times over.
    """
    pairs: List[Tuple[int, int, int, int]] = []
    for block_m in _low_waste_widths(m, 64):
        for block_n in _low_waste_widths(n, 64):
            if block_m * block_n * ACCUM_BYTES > L0C_BYTES:
                continue
            if not _k_step(block_m, block_n, k, elem_bytes):
                continue
            volume = (-(-m // block_m) * block_m
                      * -(-n // block_n) * block_n)
            # Equal volume is a real tie -- the same padded mmads either way --
            # and the wider tile settles it, having fewer programs to launch and
            # more of its operand reread on chip.
            pairs.append((volume, -(block_m * block_n), block_m, block_n))
    pairs.sort()
    return [(block_m, block_n)
            for _, _, block_m, block_n in pairs[:_MAX_DERIVED_TILINGS]]


def _wants_single_buffered_tile(m: int, n: int) -> bool:
    """Whether to offer the tile that spends the second L0C buffer on width.

    Giving up the second accumulator costs the overlap between one tile's
    fixpipe and the next tile's mmad, and that is expensive enough that a
    single-buffered tile sitting on the traffic floor still loses to the
    largest double-buffered one.  Traffic does not order these tiles; the
    buffering does.

    What buys the overlap back is removing whole passes over A.  A column of
    tiles re-reads the whole of A, so doubling the tile's width halves that
    count.  How many columns that is worth doing at was once read off a few
    very wide shapes and written down as a minimum, and the reading does not
    hold: shapes with a third of that many columns gain most of the gap to
    the library on this tile, being held up by MTE2, which is exactly what
    the halved pass relieves.  So the tile is offered wherever the output can
    fill it, and the bench is left to weigh it against the narrower ones.

    M and N must both be able to fill the tile: a 256-wide tile on a shorter
    output is padding, which pays the single-buffered price for width it
    cannot use.
    """
    return m >= WIDE_TILE_N and n >= WIDE_TILE_N


# Most L1 slots a plan may ask for.  Two, and not as a judgement: the backend
# accepts no other count, answering `only support bufferize equals 2` at
# compile time for anything else.  A plan wanting L1 staged deeper than one
# slot ahead cannot be expressed here, and is declined rather than emitted to
# fail compilation one config at a time.
L1_MAX_BUFFER_COPIES = L1_BUFFER_COPIES


@functools.lru_cache(maxsize=256)
def _solver_tiling(m: int, n: int, k: int, dtype):
    """What the operator library's own tiling solver plans for this shape.

    The table below carries powers of two and the derivation widens them to
    cover a ragged extent.  Neither reaches the tiles the library itself uses
    on a large shape, which are usually neither: it will ask for 176 x 192
    where nothing here would offer anything but 128 x 256.  mm already asks
    the solver instead of guessing at them, and this does the same.

    Cached because both the tile and the plan below want the same answer, and
    the solver searches rather than computes.
    """
    # The solver assumes fp16 when it is not told the dtype, and a tile
    # planned for two bytes is not one for four, so it is not asked at all
    # rather than asked on an assumption.
    if dtype is None:
        return None
    try:
        return get_optimized_tiling(m, n, k, dtype=dtype)
    except Exception:
        log.debug("tiling solver declined m=%s n=%s k=%s", m, n, k,
                  exc_info=True)
        return None


def _solver_tile(m: int, n: int, k: int, dtype) -> Tuple[int, int]:
    """The solver's tile, for offering with this file's own K ladder.

    Separate from the plan below because the two fail separately: a tile this
    template can hold is worth offering even when the K staging around it is
    not one the backend can express.
    """
    tiling = _solver_tiling(m, n, k, dtype)
    if tiling is None:
        return ()
    block_m, block_n = int(tiling["mL1"]), int(tiling["nL1"])
    if block_m % MMAD_M_FRACTAL or block_n % MMAD_K_FRACTAL:
        return ()
    if not 0 < block_m * block_n * ACCUM_BYTES <= L0C_BYTES:
        return ()
    return block_m, block_n


def _solver_plan(m: int, n: int, k: int, dtype, elem_bytes: int
                 ) -> Tuple[int, int, int, int]:
    """The solver's plan entire, as (BLOCK_M, BLOCK_N, BLOCK_K, copies).

    The plan is two-level in K: L1 holds kL1 and L0 is fed baseK of it at a
    time, stepKa times over.  This template has one BLOCK_K, but it also has
    the L1 slot count, and the two together say the same thing -- BLOCK_K is
    what reaches L0, and `copies` slots of it are what L1 holds.  In M and N
    the plan is single-level, the solver's L0 block equal to its L1 block on
    every shape measured, so those map straight across.

    A plan is taken whole or not at all.  Clamping one dimension of it to
    what this template can hold leaves a plan the solver never weighed, and
    its tile is offered separately anyway.
    """
    tiling = _solver_tiling(m, n, k, dtype)
    if tiling is None:
        return ()
    block_m, block_n = int(tiling["baseM"]), int(tiling["baseN"])
    block_k, copies = int(tiling["baseK"]), int(tiling["stepKa"])
    l1_k = int(tiling["kL1"])

    # M and N have to be single-level for the mapping to hold, and K's two
    # levels have to be the ones the slot count can express.
    if (block_m, block_n) != _solver_tile(m, n, k, dtype):
        return ()
    if block_k * copies != l1_k:
        return ()
    if block_k % MMAD_K_FRACTAL and block_k != k:
        return ()
    if not 0 < copies <= L1_MAX_BUFFER_COPIES:
        return ()
    # L0A holds one A block and L0B one B block, both the same size.
    if max(block_m, block_n) * block_k * elem_bytes > L0A_BYTES:
        return ()
    if copies * (block_m + block_n) * block_k * elem_bytes > L1_BYTES:
        return ()
    # One slot is not a depth the template can ask for.  L1_COPIES reaches the
    # backend as extension.multibuffer, which takes two and nothing else, so a
    # plan holding a single chunk of K has to say that it is not staged rather
    # than that it is staged once -- which is what it means, and what zero says
    # here.  Left as one the config compiles to an error and is dropped, which
    # costs the shape the plan without saying so.
    return block_m, block_n, block_k, copies if copies > 1 else 0


def _tile_shapes(m: int, n: int, k: int, elem_bytes: int, dtype=None
                 ) -> List[Tuple[int, int]]:
    """The (BLOCK_M, BLOCK_N) tiles worth offering for this shape.

    The tile layer every BLOCK_M x BLOCK_N template draws on: the generic
    template and T3 cut the output into the same tiles and differ only in how
    the tiles are handed to cores.

    The same tile geometries mm offers, plus the wider ones autotune picked up
    on large BMM shapes, BLOCK_N=256 for wide outputs.  Ragged shapes then add
    geometries derived from the shape itself, which the table cannot carry.
    """
    tile_shapes = [
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (64, 256),
        (128, 256),
        # The same accumulator as 128x256 turned the other way up, which wins
        # wherever N is the axis with more tiles in it.  No rule here derives
        # it; the table had no 256-tall entry at all.
        (256, 128),
        (32, 64),
        (64, 32),
        (32, 32),
    ]

    # The whole of L0C in one tile, at the cost of the second accumulator
    # buffer.  Where the lost overlap does not pay it is not offered at all,
    # rather than left for the bench to reject.
    if _wants_single_buffered_tile(m, n):
        tile_shapes.append((WIDE_TILE_N, WIDE_TILE_N))

    for tiling in _derived_tilings(m, n, k, elem_bytes):
        if tiling not in tile_shapes:
            tile_shapes.append(tiling)

    # Drop the tilings that cannot win on traffic.  A tile narrower than the
    # shape re-reads the operands once per tile of the other axis, and the
    # table carries widths down to 32 for the small shapes, so on a large one
    # it offers plans moving several times the bytes of the best plan in it.
    #
    # They are dropped rather than ranked below, because autotune's bench
    # does not reliably separate them: on a large shape its top candidates
    # come back within a fraction of a percent of each other and the winner
    # is decided by which tie the sort saw first, which has picked plans
    # markedly slower than the ones they displaced.
    def _traffic(tiling):
        block_m, block_n = tiling
        return (-(-n // block_n) * m * k    # A, re-read once per N tile
                + -(-m // block_m) * k * n  # B, re-read once per M tile
                + m * n)                    # C, written once

    # Twice the floor was too loose: it admitted the short tilings the ragged
    # derivation produces, which re-read one operand many times over, and the
    # bench preferred one of them to the tiling that actually profiles
    # fastest.  Shapes whose traffic floor is flat keep their spread
    # regardless, a tenth of the table usually falling inside 1.45x.
    floor = min(_traffic(t) for t in tile_shapes)
    near = [t for t in tile_shapes if _traffic(t) <= 1.45 * floor]
    # Keep a spread for autotune to choose from where the floor is sharp.
    tile_shapes = near if len(near) >= 3 else sorted(tile_shapes,
                                                     key=_traffic)[:3]

    # The solver's tile is added after the pruning rather than before it: the
    # traffic argument above weighs tiles that this file derived against each
    # other, and the solver weighed its own against the same hardware with
    # more of it in view.  Only its geometry is taken here, to be offered
    # with the K ladder like any other tile; the plan entire is a config of
    # its own, added by the generators.
    solver_tile = _solver_tile(m, n, k, dtype)
    if solver_tile and solver_tile not in tile_shapes:
        tile_shapes.append(solver_tile)
    return tile_shapes


def _rebalanced_n(batch: int, m_tiles: int, n: int, block_n: int) -> int:
    """A narrower BLOCK_N that fills the last wave better, or block_n itself.

    Cutting N finer is the one lever on the tile count that leaves M and the
    K reduction alone.  The width is still a whole number of fractals and never
    narrower than one, and it is taken only where it raises the share of the
    core-cycles spent on tiles; among equal shares the wider tile wins, having
    less of its operands to re-read.
    """
    n_tiles = -(-n // block_n)
    best, best_use = block_n, _wave_use(batch * m_tiles * n_tiles)
    for tiles in range(n_tiles + 1, 2 * n_tiles + 1):
        width = -(-n // tiles)
        width = -(-width // MMAD_K_FRACTAL) * MMAD_K_FRACTAL
        if width < MMAD_K_FRACTAL:
            break
        use = _wave_use(batch * m_tiles * -(-n // width))
        if use > best_use + 1e-9:
            best, best_use = width, use
    return best


# ---------------------------------------------------------------------------
# Generic BMM template: one program per (batch, M tile, N tile)
# ---------------------------------------------------------------------------
# The mm template plus a batch dimension.  Always offered, so there is a Triton
# choice for any shape, and {{store_output}} keeps it open to epilogue fusion.

@SymbolicGridFn
def npu_bmm_grid(b, m, n, meta, *, cdiv):
    """Grid for the generic template: one program per (batch, M tile, N tile).

    The three axes are linearised into one dimension with the batch as the most
    significant digit, rather than left as (tiles, batch): the traversal order
    reorders a *run* of tile indices, and on its own grid axis the batch would
    restart that run.
    """
    return (b * cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


_BMM_TEMPLATE = """{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # The batch is the most significant digit of the tile index, so one linear
    # run of tiles covers the whole output.
    tiles_per_batch = grid_m * grid_n
    idx_q = pid // tiles_per_batch
    tile = pid - idx_q * tiles_per_batch

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = tile // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile % group_size)
    pid_n = (tile % width) // group_size
    # A ragged tail is pulled back to end on the extent rather than masked;
    # see SHIFT_M/MASK_M in _get_npu_bmm_configs.  The clamp is on the scalar
    # base and not on rm, a tl.minimum on the vector costing the whole 2-D
    # index tensor in UB.
{% if SHIFT_M %}
    rm = tl.minimum(pid_m * BLOCK_M, M - BLOCK_M) + tl.arange(0, BLOCK_M)
{% else %}
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
{% endif %}
{% if SHIFT_N %}
    rn = tl.minimum(pid_n * BLOCK_N, N - BLOCK_N) + tl.arange(0, BLOCK_N)
{% else %}
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
{% endif %}

    {% if MASK_M %}
    m_mask = rm < M
    {% endif %}
    {% if MASK_N %}
    n_mask = rn < N
    {% endif %}

    # batch offsets precomputed before the K-loop, to avoid a redundant
    # multiply inside the hot loop
    a_batch_off = idx_q * stride_aq
    b_batch_off = idx_q * stride_bq

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Whole K steps only; a ragged K overhangs on its last step alone, which is
    # peeled out below rather than masking every iteration.
    for k_start in range(0, K_MAIN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak
                      + a_batch_off)
        b_ptrs = B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn
                      + b_batch_off)
        a = tl.load(a_ptrs{% if MASK_M %}, mask=m_mask[:, None], other=0.0{% endif %})
        b = tl.load(b_ptrs{% if MASK_N %}, mask=n_mask[None, :], other=0.0{% endif %})
{% if L1_COPIES %}
        # A second L1 slot per operand, so iteration k+1's MTE2 runs under
        # iteration k's MAC instead of waiting on the same buffer.
        extension.multibuffer(a, L1_COPIES)
        extension.multibuffer(b, L1_COPIES)
{% endif %}
        acc = tl.dot(a, b, acc=acc{% if ALLOW_HF32 %}, input_precision="tf32"{% endif %}, out_dtype=ACC_TYPE)

    {% if not EVEN_K %}
    # The peeled tail, K_MAIN .. K, which runs once and unconditionally.
    offs_k = K_MAIN + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    a_ptrs = A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak
                  + a_batch_off)
    b_ptrs = B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn
                  + b_batch_off)
    a = tl.load(
        a_ptrs,
        mask={% if MASK_M %}m_mask[:, None] & {% endif %}k_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptrs,
        mask=k_mask[:, None]{% if MASK_N %} & n_mask[None, :]{% endif %},
        other=0.0,
    )
    acc = tl.dot(a, b, acc=acc{% if ALLOW_HF32 %}, input_precision="tf32"{% endif %}, out_dtype=ACC_TYPE)
    {% endif %}

    # rematerialize rm and rn to save registers, on the same bases the loads
    # used so a pulled-back tile stores where it read
{% if SHIFT_M %}
    rm = tl.minimum(pid_m * BLOCK_M, M - BLOCK_M) + tl.arange(0, BLOCK_M)
{% else %}
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
{% endif %}
{% if SHIFT_N %}
    rn = tl.minimum(pid_n * BLOCK_N, N - BLOCK_N) + tl.arange(0, BLOCK_N)
{% else %}
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
{% endif %}
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    # Only an axis that is neither whole nor pulled back still overhangs; a
    # tile that overhangs nowhere wants no mask rather than a true one.
    {% if MASK_M and MASK_N %}
    mask = (idx_m < M) & (idx_n < N)
    {% elif MASK_M %}
    mask = idx_m < M
    {% elif MASK_N %}
    mask = idx_n < N
    {% else %}
    mask = None
    {% endif %}

    # inductor generates a suffix
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
"""

npu_triton_bmm_template = _WideIndexNPUTritonTemplate(
    name="npu_triton_bmm",
    grid=npu_bmm_grid,
    source=_BMM_TEMPLATE,
    debug=False,
    compile_options=_BMM_COMPILE_OPTIONS,
)


def _get_npu_bmm_configs(
    m: int,
    n: int,
    k: int,
    elem_bytes: int = DEFAULT_ELEM_BYTES,
    batch: int = 1,
    dtype=None,
) -> List[Dict[str, Any]]:
    """Configs for the generic template: one program per tile.

    The tiles come from _tile_shapes, and a short ladder of K depths is
    offered per tile.  Deriving the one deepest step that fits was tried and
    is worse: on a long K the optimum is interior to the depths a tile can
    hold, the deepest one neither the fastest nor near it.

    A dynamic shape gets no tiling, and aten.bmm runs from the operator
    library.  Kernels with the extents left symbolic were tried on device and
    are not safe to offer: the same plan was exact on one input and wrong on
    the next, and some tiles failed to compile at all.
    """
    # The extents arrive as sympy Integers, whose // and % return sympy rather
    # than int and reach the rendered kernel as such.  A symbolic extent, which
    # is what a dynamic shape brings here, cannot be converted at all.
    try:
        m, n, k = int(m), int(n), int(k)
    except TypeError:
        return []

    def config(block_m: int, block_n: int, block_k: int, group_m: int,
               l1_copies: int) -> Dict[str, Any]:
        return {
            # The tile one program owns, and the K step it stages in.
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            # How this tile wants L0C buffered.
            L0C_BUFFERING_KEY: _l0c_buffering(
                block_m * block_n * ACCUM_BYTES),
            # Pipeline depth and warp count, which the cube path does not
            # vary.  Four warps was measured against two and lost or tied
            # everywhere it was tried.
            "num_stages": 2,
            "num_warps": 4,
            # Whether the mmad may take the tf32 path, and the accumulator
            # type, fp32 whatever the operands are.
            "ALLOW_HF32": matmul.allow_hf32,
            "ACC_TYPE": "tl.float32",
            # Whether K divides its step.  A K that does not leaves one short
            # turn over, peeled out of the loop and masked there.
            "EVEN_K": k % block_k == 0,
            # Copies of the A/B staging buffers, 0 for none.
            "L1_COPIES": l1_copies,
            # How a ragged M or N is covered.  An extent of at least one
            # whole tile pulls its last tile back to end on the extent and
            # needs no mask on load or store; a shorter one has nowhere to
            # pull back to and is masked.  An extent that divides is neither.
            "SHIFT_M": (m % block_m != 0) and m >= block_m,
            "SHIFT_N": (n % block_n != 0) and n >= block_n,
            "MASK_M": (m % block_m != 0) and m < block_m,
            "MASK_N": (n % block_n != 0) and n < block_n,
            # K rounded down to whole steps: where the unmasked K loop stops
            # and the peeled tail picks up.  Equal to K when the step
            # divides, which leaves the tail unemitted.
            "K_MAIN": (k // block_k) * block_k,
        }

    configs: List[Dict[str, Any]] = []
    tile_shapes = _tile_shapes(m, n, k, elem_bytes, dtype)

    for block_m, block_n in tile_shapes:
        for block_k in _k_steps(block_m, block_n, k, elem_bytes):
            # GROUP_M: how many M tiles are grouped before advancing N, which
            # is the traversal order and so the L2 reuse.  Row-major (1) is
            # offered only on large tiles, where grid_m is small enough for
            # grouping to buy nothing.
            if block_m >= 128 and block_n >= 128:
                group_m_values = [1, 8]
            else:
                group_m_values = [8]
            # Whether a second L1 slot per operand fits beside the first, which
            # is what lets the next K step's MTE2 run under this step's MAC.
            # Both are offered: the slot is worth a tile of L1 on a long K and
            # worth nothing on a K of one step.
            staged = 2 * (block_m + block_n) * block_k * elem_bytes
            l1_copies_values = ([0, L1_BUFFER_COPIES]
                                if staged <= L1_BYTES and k > block_k else [0])
            for group_m, l1_copies in itertools.product(group_m_values,
                                                        l1_copies_values):
                configs.append(
                    config(block_m, block_n, block_k, group_m, l1_copies))

    # The library's plan entire, with the traversal order left to the same
    # choice the table's large tiles get.
    plan = _solver_plan(m, n, k, dtype, elem_bytes)
    if plan:
        block_m, block_n, block_k, l1_copies = plan
        group_m = 1 if block_m >= 128 and block_n >= 128 else 8
        solver_config = config(block_m, block_n, block_k, group_m, l1_copies)
        if solver_config not in configs:
            configs.append(solver_config)

    return configs


# ---------------------------------------------------------------------------
# T3 tile template: one BLOCK_M x BLOCK_N tile at a time, handed out by core
# ---------------------------------------------------------------------------
# The A class: one batch fills several tiles on its own, so the output is cut
# into tiles and the batch is nothing more than the most significant digit of
# the tile index.  TOTAL_TILES = batch x M_TILES x N_TILES.
#
# Tiles go to cores in index order, the hardware handing each core the next
# index as it frees up, so a wave is one run of consecutive indices.  Those
# indices are laid over the output a window of WINDOW M tiles at a time,
# scanning N across the window and turning back on every other column.  A wave
# then covers a block of tiles that share their A rows across it and their B
# columns down it, which is what L2 can serve to the second reader; the turn
# keeps that reuse from breaking at the column boundary.
#
# K is never divided: each tile is reduced over the whole of K by the core that
# owns it, BLOCK_K at a time into one L0C accumulator, and stored once.
#
# The per-tile body is the generic template's.  It is rank 2, so this template
# does not depend on the rank-3 BatchMmadL1Op lowering T2 needs.

@SymbolicGridFn
def npu_bmm_tile_grid(b, m, n, meta, *, cdiv):
    """Grid for T3: one program per tile."""
    return (meta["TOTAL_TILES"], 1, 1)


_BMM_TILE_TEMPLATE = r"""{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # One tile per program, and no guard around it: launching TOTAL_TILES
    # programs needs none, and wrapping the tile in `if tile < TOTAL_TILES:`
    # puts the whole reduction inside a run-time branch, which measured
    # slower than any traversal order bought back.
    tile = tl.program_id(0).to(INDEX_DTYPE)
    idx_q = tile // TILES_PER_BATCH
    rest = tile - idx_q * TILES_PER_BATCH
    # The window this tile falls in, and its height: the last window of a
    # batch is short when WINDOW does not divide M_TILES.
    first_m = (rest // (WINDOW * N_TILES)) * WINDOW
    height = tl.minimum(M_TILES - first_m, WINDOW)
    in_window = rest - first_m * N_TILES
    pid_n = in_window // height
    down = in_window - pid_n * height
    # Odd columns are walked upwards.
    pid_m = first_m + down + (pid_n % 2) * (height - 1 - 2 * down)

    # A ragged M or N is pulled back to end on the extent rather than masked,
    # as in the generic template.
{% if SHIFT_M %}
    m_base = tl.minimum(pid_m * BLOCK_M, M - BLOCK_M)
{% else %}
    m_base = pid_m * BLOCK_M
{% endif %}
{% if SHIFT_N %}
    n_base = tl.minimum(pid_n * BLOCK_N, N - BLOCK_N)
{% else %}
    n_base = pid_n * BLOCK_N
{% endif %}
    rm = m_base + tl.arange(0, BLOCK_M)
    rn = n_base + tl.arange(0, BLOCK_N)
{% if MASK_M %}
    m_mask = rm < M
{% endif %}
{% if MASK_N %}
    n_mask = rn < N
{% endif %}
    a_batch_off = idx_q * stride_aq
    b_batch_off = idx_q * stride_bq

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k_start in range(0, K_MAIN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak
                      + a_batch_off)
        b_ptrs = B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn
                      + b_batch_off)
        a = tl.load(a_ptrs{% if MASK_M %}, mask=m_mask[:, None], other=0.0{% endif %})
        b = tl.load(b_ptrs{% if MASK_N %}, mask=n_mask[None, :], other=0.0{% endif %})
{% if L1_COPIES %}
        extension.multibuffer(a, L1_COPIES)
        extension.multibuffer(b, L1_COPIES)
{% endif %}
        acc = tl.dot(a, b, acc=acc{% if ALLOW_HF32 %}, input_precision="tf32"{% endif %}, out_dtype=ACC_TYPE)
{% if not EVEN_K %}
    # The peeled tail, K_MAIN .. K, masked on K and zero-filled.
    offs_k = K_MAIN + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    a_ptrs = A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak
                  + a_batch_off)
    b_ptrs = B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn
                  + b_batch_off)
    a = tl.load(
        a_ptrs,
        mask={% if MASK_M %}m_mask[:, None] & {% endif %}k_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptrs,
        mask=k_mask[:, None]{% if MASK_N %} & n_mask[None, :]{% endif %},
        other=0.0,
    )
    acc = tl.dot(a, b, acc=acc{% if ALLOW_HF32 %}, input_precision="tf32"{% endif %}, out_dtype=ACC_TYPE)
{% endif %}

    idx_m = rm[:, None]
    idx_n = rn[None, :]
{% if MASK_M and MASK_N %}
    mask = (idx_m < M) & (idx_n < N)
{% elif MASK_M %}
    mask = idx_m < M
{% elif MASK_N %}
    mask = idx_n < N
{% else %}
    mask = None
{% endif %}
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask",
                   val_shape=("BLOCK_M", "BLOCK_N"))}}
"""

npu_triton_bmm_tile_template = _WideIndexNPUTritonTemplate(
    name="npu_triton_bmm_tile",
    grid=npu_bmm_tile_grid,
    source=_BMM_TILE_TEMPLATE,
    debug=False,
    compile_options=_BMM_COMPILE_OPTIONS,
)


# The operand dtypes T3 is offered at.  It reaches fp32, unlike the rank-3
# templates, but it is still a float path: the accumulator is fp32 and the
# mmad is given a tf32 hint, neither of which carries integer semantics.
_TILE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _get_npu_bmm_tile_configs(
    batch: int,
    m: int,
    n: int,
    k: int,
    elem_bytes: int = DEFAULT_ELEM_BYTES,
    dtype=None,
) -> List[Dict[str, Any]]:
    """Configs for T3, the template that hands tiles out by core.

    The tiles are _tile_shapes', the ones the generic template offers.  Each
    is offered as it is and, where its last wave is short, once more with N cut
    finer to fill it.  Each then gets the K staging depths _k_steps allows and
    the window decides the order; nothing here divides K.

    The library's own plan is offered too, entire: its tile, its L0 depth and
    the L1 slots that hold the rest of its L1 depth.
    """
    try:
        batch, m, n, k = int(batch), int(m), int(n), int(k)
    except TypeError:
        return []
    if batch <= 0 or m <= 0 or n <= 0 or k <= 0:
        return []

    def config(block_m: int, block_n: int, block_k: int,
               l1_copies: int) -> Dict[str, Any]:
        m_tiles = -(-m // block_m)
        n_tiles = -(-n // block_n)
        tiles_per_batch = m_tiles * n_tiles
        return {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            # The tile walk.  TOTAL_TILES, the batch times the tiles of one
            # batch, is also the program count.
            "M_TILES": m_tiles,
            "N_TILES": n_tiles,
            "TILES_PER_BATCH": tiles_per_batch,
            "TOTAL_TILES": batch * tiles_per_batch,
            "WINDOW": _window(m_tiles),
            # Tails: pulled back where the extent holds one whole tile,
            # masked where it does not, neither where the tile divides.
            "SHIFT_M": m % block_m != 0 and m >= block_m,
            "SHIFT_N": n % block_n != 0 and n >= block_n,
            "MASK_M": m % block_m != 0 and m < block_m,
            "MASK_N": n % block_n != 0 and n < block_n,
            "EVEN_K": k % block_k == 0,
            "K_MAIN": (k // block_k) * block_k,
            "L1_COPIES": l1_copies,
            L0C_BUFFERING_KEY: _l0c_buffering(
                block_m * block_n * ACCUM_BYTES),
            "num_stages": 2,
            "num_warps": 4,
            "ALLOW_HF32": matmul.allow_hf32,
            "ACC_TYPE": "tl.float32",
        }

    tilings: List[Tuple[int, int]] = []
    for block_m, block_n in _tile_shapes(m, n, k, elem_bytes, dtype):
        m_tiles = -(-m // block_m)
        for width in (block_n, _rebalanced_n(batch, m_tiles, n, block_n)):
            if (block_m, width) not in tilings:
                tilings.append((block_m, width))

    configs: List[Dict[str, Any]] = []
    for block_m, block_n in tilings:
        # The whole of L0C, not half: a tile past half of it is offered
        # single-buffered rather than dropped.
        if block_m * block_n * ACCUM_BYTES > L0C_BYTES:
            continue
        for block_k in _k_steps(block_m, block_n, k, elem_bytes):
            staged = L1_BUFFER_COPIES * (block_m + block_n) * block_k * elem_bytes
            # Both operands double-buffered where the pair fits L1 twice
            # and K takes more than one step; otherwise one slot each.
            configs.append(config(
                block_m, block_n, block_k,
                L1_BUFFER_COPIES if staged <= L1_BYTES and k > block_k else 0,
            ))

    plan = _solver_plan(m, n, k, dtype, elem_bytes)
    if plan:
        solver_config = config(*plan)
        if solver_config not in configs:
            configs.append(solver_config)
    return configs


# ---------------------------------------------------------------------------
# T1 merged-batch template: G micro-batches folded into one fractal-sized mmad
# ---------------------------------------------------------------------------
# The operand dtypes T1 and T2 are offered at.  Both lower to a rank-3 tl.dot,
# which is exact at two-byte floats and neither exact nor safe above them.
_RANK3_DTYPES = (torch.float16, torch.bfloat16)


# The C class: one batch's output is narrower than a fractal on both sides, so
# any template that gives each batch its own mmad pads it to 16 x 16 and the
# padding is nearly all of the work -- at M=N=1 one output in 256.
#
# G consecutive batches are taken together.  With the batch contiguous in GM,
# their A rows are one (G*M) x K block and their B columns side by side are
# K x (G*N).  One mmad of the two gives every (row, column) pair; the pairs
# whose row and column come from the same batch are the block diagonal, and
# those are the outputs, the rest being computed and never stored.  Under a
# fractal that costs nothing: the mmad was going to be one fractal wide
# anyway, and now G batches fill it.
#
# K is never divided: BLOCK_K is the whole of K rounded up to a fractal, the
# overhang loaded as zero.  Groups are walked by programs as in T2, and the
# last group is pulled back to end on the batch rather than masked.

@SymbolicGridFn
def npu_bmm_merged_grid(b, m, n, meta, *, cdiv):
    """Grid for T1: one program per run of groups."""
    return (meta["PROGRAMS"], 1, 1)


_BMM_MERGED_TEMPLATE = r"""{{def_kernel("A", "B")}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # Row r of the folded A is row r % M of batch r // M, column c of the
    # folded B is column c % N of batch c // N.  The pad past G*M rows and G*N
    # columns rereads the last real one; it is never stored.
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    row = tl.minimum(rows, G * M_EXT - 1)
    col = tl.minimum(cols, G * N_EXT - 1)
    a_batch = row // M_EXT
    b_batch = col // N_EXT
    a_m = row - a_batch * M_EXT
    b_n = col - b_batch * N_EXT
    offs_k = tl.arange(0, BLOCK_K)
    diagonal = ((rows < G * M_EXT)[:, None] & (cols < G * N_EXT)[None, :]
                & (a_batch[:, None] == b_batch[None, :]))

    first_group = tl.program_id(0) * GROUPS_PER_PROGRAM
{% if RAGGED_GROUPS %}
    n_groups = tl.maximum(0, tl.minimum(GROUPS_PER_PROGRAM, TOTAL_GROUPS - first_group))
{% else %}
    n_groups = GROUPS_PER_PROGRAM
{% endif %}
    for g in range(n_groups):
{% if SHIFT_Q %}
        q0 = tl.minimum((first_group + g) * G, BATCH - G)
{% else %}
        q0 = (first_group + g) * G
{% endif %}
        idx_qa = (q0 + a_batch).to(INDEX_DTYPE)
        idx_qb = (q0 + b_batch).to(INDEX_DTYPE)
        a_ptrs = (A + idx_qa[:, None] * stride_aq + a_m[:, None] * stride_am
                  + offs_k[None, :] * stride_ak)
        b_ptrs = (B + offs_k[:, None] * stride_bk + idx_qb[None, :] * stride_bq
                  + b_n[None, :] * stride_bn)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.zeros((ROWS, COLS), dtype=ACC_TYPE)
        acc = tl.dot(a, b, acc=acc{% if ALLOW_HF32 %}, input_precision="tf32"{% endif %}, out_dtype=ACC_TYPE)

        idx_q = idx_qa[:, None]
        idx_m = a_m[:, None]
        idx_n = b_n[None, :]
        mask = diagonal
        # No val_shape.  Given one, inductor checks every epilogue load against
        # it and reads idx_q -- a column here, not the scalar the other
        # templates pass -- as one XBLOCK-long vector it cannot broadcast
        # against (ROWS, COLS), so the compile fails.  Triton broadcasts the
        # three indices itself, so the check has nothing to catch.
        {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask",
                       indent_width=8)}}
"""

npu_triton_bmm_merged_template = _WideIndexNPUTritonTemplate(
    name="npu_triton_bmm_merged",
    grid=npu_bmm_merged_grid,
    source=_BMM_MERGED_TEMPLATE,
    debug=False,
    compile_options=_BMM_COMPILE_OPTIONS,
)


def _merged_group(m: int, n: int) -> int:
    """How many batches T1 folds into one mmad, or 0 where folding gains nothing.

    The cost of a group is the fractals its folded mmad covers,
    ceil(G*M/16) x ceil(G*N/16), against the one fractal per batch that any
    per-batch mmad pays at these sizes; the smallest G that brings the cost
    per batch to its minimum is taken, the larger ones only computing more of
    the discarded off-diagonal.  A power of two, so the group walk is a shift.
    """
    def per_batch(g):
        return (-(-g * m // MMAD_M_FRACTAL) * -(-g * n // MMAD_K_FRACTAL)) / g

    best, best_cost = 0, 1.0
    g = 2
    while g * max(m, n) <= 2 * MMAD_M_FRACTAL:
        cost = per_batch(g)
        if cost < best_cost - 1e-9:
            best, best_cost = g, cost
        g *= 2
    return best


def _get_npu_bmm_merged_configs(
    batch: int,
    m: int,
    n: int,
    k: int,
    elem_bytes: int = DEFAULT_ELEM_BYTES,
) -> List[Dict[str, Any]]:
    """Configs for T1, the merged-batch template.

    Offered on the C class of the design doc: the batch is many times the core
    count, K is at least four fractals, and one batch's output is under a
    fractal on both sides.  That last test is narrower than the doc's "under one
    tile": a batch whose M or N reaches a fractal already fills the mmad along
    that side, and folding it by the block diagonal only multiplies the
    discarded work.  Those shapes are T2's, whose rank-3 mmad is the per-batch
    form of the same merge.

    The caller has checked that both operands carry a batch and are contiguous
    within it, which is what makes G batches of A one 2-D block.
    """
    try:
        batch, m, n, k = int(batch), int(m), int(n), int(k)
    except TypeError:
        return []
    if batch < 4 * NUM_CUBE_CORES:
        return []
    # K has to be whole fractals already; padding and masking it does not hold
    # here any more than it does in T2, see the note there.  These shapes go
    # to T3, which peels a tail instead.
    if k % MMAD_K_FRACTAL:
        return []
    block_k = k
    if block_k < 4 * MMAD_K_FRACTAL:
        return []
    if m >= MMAD_M_FRACTAL or n >= MMAD_K_FRACTAL or m < 1 or n < 1:
        return []
    # N of one leaves the output index constant along the column axis, every
    # column of the block diagonal addressing the same element, and the store
    # does not survive it: only one lane per group is written and the rest of
    # the output stays zero.  The same shapes with N above one are exact, so
    # it is the column-constant index and not the fold.  T2 answers them
    # instead, and faster.
    if n < 2:
        return []
    group = _merged_group(m, n)
    if not group or group > batch:
        return []
    rows = -(-group * m // MMAD_M_FRACTAL) * MMAD_M_FRACTAL
    cols = -(-group * n // MMAD_K_FRACTAL) * MMAD_K_FRACTAL
    if rows * cols * ACCUM_BYTES > L0C_DOUBLE_BUFFERED_BYTES:
        return []
    if L1_BUFFER_COPIES * (rows + cols) * block_k * elem_bytes > L1_BYTES:
        return []
    if rows * block_k * elem_bytes > L0A_BYTES:
        return []

    total_groups = -(-batch // group)
    programs, groups_per_program, ragged = _split_groups(total_groups)
    return [{
        "G": group,
        "M_EXT": m,
        "N_EXT": n,
        "BATCH": batch,
        "ROWS": rows,
        "COLS": cols,
        "BLOCK_K": block_k,
        "TOTAL_GROUPS": total_groups,
        "GROUPS_PER_PROGRAM": groups_per_program,
        "RAGGED_GROUPS": ragged,
        "PROGRAMS": programs,
        "SHIFT_Q": batch % group != 0,
        L0C_BUFFERING_KEY: _l0c_buffering(rows * cols * ACCUM_BYTES),
        "num_stages": 2,
        "num_warps": 4,
        "ALLOW_HF32": matmul.allow_hf32,
        "ACC_TYPE": "tl.float32",
    }]


# ---------------------------------------------------------------------------
# T2 batched template: BLOCK_Q batches per rank-3 mmad
# ---------------------------------------------------------------------------
# The grouped regime: many batches, each with a working set small enough that
# one batch's A and B sit in L1 together.  Cores divide the batch and nothing
# else.  One program walks GROUPS_PER_PROGRAM consecutive groups, each a rank-3
# tl.dot over BLOCK_Q batches, which lowers to BatchMmadL1Op.  The group loop is
# an scf.for rather than separate program instances so iteration i+1's MTE2
# runs under iteration i's MAC, and extension.multibuffer marks the L1 staging
# buffers so the compiler double-buffers across groups.
#
# Inside a group the output is cut into M_TILES x N_TILES tiles only as far as
# the accumulator needs to fit L0C; every tile reduces the whole of K in one
# mmad, so no partial sum ever leaves the core.

@SymbolicGridFn
def npu_bmm_batched_grid(b, m, n, meta, *, cdiv):
    """Grid for the batched template: one program per run of batch groups."""
    return (meta["N_GROUP"], 1, 1)


_BMM_BATCHED_TEMPLATE = r"""{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}
    BATCH = {{size("A", 0)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    offs_m0 = tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_q = tl.arange(0, BLOCK_Q)

    # Contiguous rather than strided by the program count, so one program walks
    # one slab of A and B instead of revisiting the whole tensor
    # GROUPS_PER_PROGRAM times.
    first_group = tl.program_id(0) * GROUPS_PER_PROGRAM

    # Every group is one nd2nz -> nd2nz -> mmad -> fixpipe chain with no slack
    # in it; only a neighbouring group can fill the idle pipes, and only if the
    # compiler can see it, which means it has to be a loop body.
{% if RAGGED_GROUPS %}
    # The last run comes up short.  This is a trip count and nothing else,
    # with no element mask anywhere in the body.
    n_groups = tl.maximum(0, tl.minimum(GROUPS_PER_PROGRAM, TOTAL_GROUPS - first_group))
{% else %}
    n_groups = GROUPS_PER_PROGRAM
{% endif %}
    for g in range(n_groups):
{% if SHIFT_Q %}
        # BLOCK_Q does not divide the batch, so the last group is pulled back
        # to end at the batch rather than masked, a batch mask costing every
        # load a linalg.fill.
        q = tl.minimum((first_group + g) * BLOCK_Q, BATCH - BLOCK_Q) + offs_q
{% else %}
        q = (first_group + g) * BLOCK_Q + offs_q
{% endif %}
        # In the index dtype, not q's: every offset below scales with the
        # batch, and past 2^31 elements an int32 product wraps.
        idx_q = q[:, None, None].to(INDEX_DTYPE)

        # K is whole fractals already, checked before this config was offered,
        # and never split: BLOCK_K is all of it and no load masks it.
        for m_tile in range(M_TILES):
            offs_m = m_tile * BLOCK_M + offs_m0
            a_ptrs = A + idx_q * stride_aq + offs_m[None, :, None] * stride_am + offs_k[None, None, :] * stride_ak
{% if MASKED_M %}
            a = tl.load(a_ptrs, mask=offs_m[None, :, None] < M, other=0.0)
{% else %}
            a = tl.load(a_ptrs)
{% endif %}
{% if L1_COPIES %}
            extension.multibuffer(a, L1_COPIES)
{% endif %}

            for n_tile in range(N_TILES):
                offs_n = n_tile * BLOCK_N + offs_n0
                b_ptrs = B + idx_q * stride_bq + offs_k[None, :, None] * stride_bk + offs_n[None, None, :] * stride_bn
{% if MASKED_N %}
                b = tl.load(b_ptrs, mask=offs_n[None, None, :] < N, other=0.0)
{% else %}
                b = tl.load(b_ptrs)
{% endif %}
{% if L1_COPIES %}
                extension.multibuffer(b, L1_COPIES)
{% endif %}

                acc = tl.zeros((BLOCK_Q, BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
                acc = tl.dot(a, b, acc=acc{% if ALLOW_HF32 %}, input_precision="tf32"{% endif %}, out_dtype=ACC_TYPE)

                idx_m = offs_m[None, :, None]
                idx_n = offs_n[None, None, :]
{% if MASKED_M and MASKED_N %}
                mask = (idx_m < M) & (idx_n < N)
{% elif MASKED_M %}
                mask = idx_m < M
{% elif MASKED_N %}
                mask = idx_n < N
{% else %}
                mask = None
{% endif %}
                {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask",
                               val_shape=("BLOCK_Q", "BLOCK_M", "BLOCK_N"),
                               indent_width=16)}}
"""

npu_triton_bmm_batched_template = _WideIndexNPUTritonTemplate(
    name="npu_triton_bmm_batched",
    grid=npu_bmm_batched_grid,
    source=_BMM_BATCHED_TEMPLATE,
    debug=False,
    compile_options=_BMM_COMPILE_OPTIONS,
)

# N fragment width of BatchL1Mmad for the two-byte operand regime below.  The M
# and K widths come from mm_common as MMAD_M_FRACTAL and MMAD_K_FRACTAL; there
# is no N entry there, this template being the only caller that needs one.
BATCHED_N_FRACTAL = 16


def _get_npu_bmm_batched_configs(
    batch: int,
    m: int,
    n: int,
    k: int,
    elem_bytes: int = DEFAULT_ELEM_BYTES,
) -> List[Dict[str, Any]]:
    """Configs for the batched template: many batches, each one small.

    Offered when the batch outnumbers the cores and one batch's A and B,
    double-buffered, fit L1.  Everything else follows from the capacities
    rather than from a shape table:

      * M, N and K are rounded up to whole fractals, not to powers of two.
      * K is never split: BLOCK_K is the whole of K, its pad zeroed.
      * The output is cut into tiles only as far as the accumulator needs to
        fit L0C, along N first: A is the operand held across the N tiles, so
        keeping its rows whole keeps it to one load per group.
      * BLOCK_Q is as many batches as L1 and L0C hold, capped so every core
        still gets MIN_WAVES groups.  It need not divide the batch.
    """
    batch, m, n, k = int(batch), int(m), int(n), int(k)

    if batch <= NUM_CUBE_CORES:
        return []

    # K has to be whole fractals already.  Where it is not, this template
    # pads it and masks the pad away on both operands, and the mask does not
    # hold: the pad reaches the mmad and the output comes back nan.  It only
    # shows when something else has left the staging buffers dirty -- clean
    # run alone, nan with a second kernel on the device -- so the zero it
    # reads is L1 residue rather than anything the mask put there.  T3
    # reduces K with a peeled tail instead of a pad and has no such
    # dependence, so these shapes go there.
    if k % MMAD_K_FRACTAL:
        return []

    def align(extent, fractal):
        return -(-extent // fractal) * fractal

    whole_m = align(m, MMAD_M_FRACTAL)
    whole_n = align(n, BATCHED_N_FRACTAL)
    block_k = align(k, MMAD_K_FRACTAL)

    # MarkMultiBuffer double-buffers the L1 staging buffers, so budget for
    # every copy.
    if (L1_BUFFER_COPIES * (whole_m + whole_n) * block_k * elem_bytes
            > L1_BYTES):
        return []

    # The K pad stops at the fractal.  Padding K by whole fractals past it,
    # as the power-of-two rounding did, left them unzeroed under the load
    # mask and the output came back wrong, or nan, once an epilogue fused.
    m_tiles = n_tiles = 1
    block_m, block_n = whole_m, whole_n
    while block_m * block_n * ACCUM_BYTES > L0C_DOUBLE_BUFFERED_BYTES:
        if block_n > BATCHED_N_FRACTAL:
            n_tiles += 1
            block_n = align(-(-whole_n // n_tiles), BATCHED_N_FRACTAL)
        elif block_m > MMAD_M_FRACTAL:
            m_tiles += 1
            block_m = align(-(-whole_m // m_tiles), MMAD_M_FRACTAL)
        else:
            return []
    m_tiles = -(-m // block_m)
    n_tiles = -(-n // block_n)

    # Capacity is not the binding constraint -- the group loop is.  One group
    # is a single nd2nz -> nd2nz -> mmad -> fixpipe chain with nothing to
    # overlap against; MIN_WAVES groups per core reproduces the knee where
    # overlap stops paying for per-group fixed cost.
    l1_per_batch = L1_BUFFER_COPIES * (block_m + block_n) * block_k * elem_bytes
    block_q = min(L1_BYTES // l1_per_batch,
                  L0C_DOUBLE_BUFFERED_BYTES
                  // (block_m * block_n * ACCUM_BYTES),
                  max(1, batch // (MIN_WAVES * NUM_CUBE_CORES)))

    # Folding several batches into one mmad is the whole of what this template
    # has over the ones that tile the output, so a group of one batch is not
    # worth offering: it is a tiled kernel with none of the tiling choices,
    # and it loses to T3 by a wide margin.  Those shapes go to T3 instead.
    if block_q < 2:
        return []

    total_groups = -(-batch // block_q)
    programs, groups_per_program, ragged = _split_groups(total_groups)

    return [{
        # The group one mmad covers: BLOCK_Q batches of one BLOCK_M by BLOCK_N
        # tile, reduced over the whole of BLOCK_K at once.
        "BLOCK_Q": block_q,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        # How the output of one group is cut so its accumulator fits L0C.
        "M_TILES": m_tiles,
        "N_TILES": n_tiles,
        # The run of groups one program walks, and whether the last programs
        # come up short of it, which costs a trip count and no mask.
        "GROUPS_PER_PROGRAM": groups_per_program,
        "RAGGED_GROUPS": ragged,
        "TOTAL_GROUPS": total_groups,
        # Whether the last group is pulled back to end at the batch.
        "SHIFT_Q": batch % block_q != 0,
        # Grid dimension: one program per run of groups.  Stored in meta so
        # npu_bmm_batched_grid can read it without re-deriving the split.
        "N_GROUP": programs,
        # Copies of the A/B staging buffers, so the next group's nd2nz runs
        # under this group's mmad instead of waiting on the same L1 slot.
        "L1_COPIES": L1_BUFFER_COPIES,
        # Whether an axis overhangs its fractal-aligned tiles and needs masking.
        # K never does: a K that is not whole fractals is turned away above.
        "MASKED_M": m_tiles * block_m != m,
        "MASKED_N": n_tiles * block_n != n,
        # Explicitly off, not merely unset: --enable-hivm-unit-flag-sync
        # defaults to true, and the flag's disable state is device-global
        # rather than keyed by kernel, so two kernels in one process alias on
        # flag group 0 and one's Fixpipe can make the next one's mmad skip a
        # FIX->M pair it needed.  BatchL1Mmad also updates the flag only on the
        # batch's final mmad.
        "unit_flag": False,
        L0C_BUFFERING_KEY: _l0c_buffering(
            block_q * block_m * block_n * ACCUM_BYTES),
        # The depth the loops run ahead, and a warp count the cube path does
        # not vary.
        "num_stages": 4,
        "num_warps": 4,
        "ALLOW_HF32": matmul.allow_hf32,
        "ACC_TYPE": "tl.float32",
        # Not a tuning knob.  This is the only template emitting a rank-3
        # tl.dot, and without the option its loads lower to a 2-D L1 load the
        # template library has no entry for, so the kernel fails to link.
        "enable_hivm_batch_matmul": True,
    }]


# ---------------------------------------------------------------------------
# Dispatch: two gates, then the first tier that answers
# ---------------------------------------------------------------------------

# The block the dispatch gates reason in: one mmad tile of the output, the
# size the design doc's A-class example settles on.  A gate only needs a
# representative block; which tile a plan actually uses is left to the config
# generators.
_GATE_BLOCK_M = 256
_GATE_BLOCK_N = 128

# Where K stops being a reduction one program should be asked to carry alone.
# The operator library's Stream-K tiling turns on at the same length, and over
# 50336 generated shapes the gate below agreed with its choice on every one.
_LONG_K = 8192


def _carries_batch(node) -> bool:
    """Whether this operand has a batch of its own, rather than one matrix
    broadcast across one.

    The first gate of the dispatch.  A broadcast operand makes the shape an mm
    wearing a batch dimension, which none of the specialised templates holds:
    each indexes A, B and C by a batch stride that has to be there.  The
    generic template is still offered, since a shape with no candidate runs
    from the operator library and takes every epilogue fusion with it.
    """
    size, stride = node.get_size(), node.get_stride()
    return len(size) == 3 and stride[0] != 0


def _degenerate_axis(mat1, mat2) -> bool:
    """Whether M, N or K is one, and known to be so before the kernel runs.

    Only a literal one counts.  The decomposition bakes the extent into the
    kernel's indexing, and a dimension that is merely one for the sizes this
    trace happened to see is a guard it cannot make.  Anything symbolic is
    left to the templates, which carry their extents as arguments.
    """
    try:
        _, m, k = mat1.get_size()
        _, _, n = mat2.get_size()
    except (TypeError, ValueError):
        return False

    def literal(axis):
        # int() takes a sympy Integer and raises on anything still symbolic,
        # which is the test.
        try:
            return int(axis)
        except TypeError:
            return None

    return any(literal(axis) == 1 for axis in (m, n, k))


def _only_k_split_fills_cores(batch: int, m: int, n: int, k: int) -> bool:
    """Whether the cores can only be filled by splitting K.

    The second gate of the dispatch.  What gets cut up is the output: when
    batch and the two output axes together cannot hand every cube core a block
    of C, the only work left to spread is along K, and reaching it means
    splitting K.

    Splitting K is the one division no template here makes.  Each piece of a
    split K is a partial sum, and a fused epilogue would run on it as though
    it were the result, so a shape that needs the split is declined outright
    and answered by the operator library, whose Stream-K tiling does split it.

    A batch below one is the caller saying it could not read one, and the
    tile count this gate turns on cannot be formed without it.  Declining is
    the destructive answer -- it takes every candidate away from the shape,
    the generic template included -- so an unknown batch does not decline.
    Nothing here is unsafe to offer on such a shape: no template below splits
    K, so the worst an unknown batch costs is a plan that could have been
    faster, where declining on it costs the shape its only plan.
    """
    if batch < 1:
        return False
    tiles = batch * -(-m // _GATE_BLOCK_M) * -(-n // _GATE_BLOCK_N)
    return tiles < NUM_CUBE_CORES and k >= _LONG_K


def add_npu_triton_bmm_choices(
    choices: List[ir.ChoiceCaller],
    layout: "ir.Layout",
    mat1: "ir.IRNode",
    mat2: "ir.IRNode",
    m: int,
    n: int,
    k: int,
    batch: int = 0,
) -> None:
    """Offer the NPU Triton bmm templates for this shape.

    Two gates, then the first tier that answers:

      gate 1  both operands carry a batch of their own, or the shape is an mm
              and only the generic template answers it
      gate 2  the cores can be filled without splitting K, or the shape is
              declined and left to the operator library

      T1 merged   many batches whose output is under a fractal each
      T2 batched  a small per-batch working set over many batches
      T3 tile     the rest of the static shapes, cut into tiles
      generic     the fallback, which assumes nothing and always answers

    The tiers are tried in order and the first one with configs takes the
    shape, T1 and T2 together as one tier.  Offering a later tier beside an
    earlier one costs time rather than buying it: the ranking is left to
    autotune's bench, and that bench cannot separate candidates this close.
    Offered side by side it has picked the generic template over a
    specialised one that then profiled markedly faster.

    Whether a shape belongs to T2 is decided from the on-chip capacities
    inside its config generator, so the generator's answer is the gate.
    """
    input_nodes = [mat1, mat2]

    # Every template budgets L0 or L1 in bytes, so they need the operand size.
    dtype = mat1.get_dtype()
    elem_bytes = _elem_bytes(dtype)

    # A K known only at run time cannot be put through gate 2, and a long one
    # is exactly the case that has to be declined, so the shape goes to the
    # library rather than being answered on a guess.
    try:
        int(k)
    except TypeError:
        log.info(
            "NPU Triton bmm: declining bmm(%s, %s, %s) batch=%s -- K is "
            "dynamic, so whether the shape needs K split cannot be decided.",
            m, n, k, batch,
        )
        return

    # Gate 2.  A gate that raises must not decline: it would take away the
    # shape's only candidate on the strength of a bug in the gate.
    try:
        needs_k_split = _only_k_split_fills_cores(batch, int(m), int(n), int(k))
    except (TypeError, ValueError):
        log.debug("K gate failed to evaluate", exc_info=True)
        needs_k_split = False
    if needs_k_split:
        log.info(
            "NPU Triton bmm: declining bmm(%s, %s, %s) batch=%s -- the output "
            "cannot fill the cores and K is long, so only splitting K would "
            "feed them, which would leave a fused epilogue reading partial "
            "sums.",
            m, n, k, batch,
        )
        return

    candidates: List = []

    # Gate 1.  Both operands must carry a batch of their own; a broadcast one
    # makes this an mm, which the specialised templates cannot index.  They
    # also need the batch as a number: each of them bakes its walk over the
    # batch into the kernel.
    both_batched = all(_carries_batch(node) for node in (mat1, mat2))
    specialised = bool(batch) and both_batched

    # T1 and T2 share a tier: both serve a batch of small outputs, and which
    # of the two wins is a ranking autotune does get right, the gap between
    # them being wide.
    #
    # Both emit a rank-3 tl.dot, and that path is offered at fp16 and bf16
    # only: at fp32 every rank-3 plan tried either returned wrong results or
    # faulted the AI core, while the identical shapes are exact at two bytes.
    # Autotune ranks on time alone and so can reject neither.  The gate names
    # the dtypes rather than their size, because the templates accumulate in
    # fp32 and hint the mmad at tf32, which is not what an integer operand of
    # the same width means.
    if specialised and dtype in _RANK3_DTYPES:
        merged_configs = _get_npu_bmm_merged_configs(batch, m, n, k, elem_bytes)
        if merged_configs:
            candidates.append((npu_triton_bmm_merged_template, merged_configs))

        batched_configs = _get_npu_bmm_batched_configs(batch, m, n, k, elem_bytes)
        if batched_configs:
            candidates.append(
                (npu_triton_bmm_batched_template, batched_configs)
            )

    # T3 takes the static shapes the tier above left.  It is also the one
    # specialised template open to fp32: the failures above are the rank-3
    # lowering's, while T3 is rank 2 and matched a double-precision golden on
    # the same shapes.
    if not candidates and specialised and dtype in _TILE_DTYPES:
        tile_configs = _get_npu_bmm_tile_configs(batch, m, n, k, elem_bytes,
                                                 dtype)
        if tile_configs:
            candidates.append((npu_triton_bmm_tile_template, tile_configs))

    # The fallback, for every shape no tier above claimed.
    if not candidates:
        candidates.append(
            (npu_triton_bmm_template,
             _get_npu_bmm_configs(m, n, k, elem_bytes, batch, dtype))
        )

    def offer(entries) -> int:
        """Generate every config of these templates; how many were taken."""
        taken = 0
        for template, configs in entries:
            for cfg in configs:
                cfg = dict(cfg)
                num_stages = cfg.pop("num_stages")
                num_warps = cfg.pop("num_warps")

                try:
                    choice = template.generate(
                        input_nodes=input_nodes,
                        layout=layout,
                        num_stages=num_stages,
                        num_warps=num_warps,
                        **cfg,
                    )
                    if choice is not None:
                        choices.append(choice)
                        taken += 1
                except Exception:
                    log.debug(
                        "Failed to generate %s choice with config %s",
                        template.name,
                        cfg,
                        exc_info=True,
                    )
        return taken

    offered = offer(candidates)
    # A tier claiming the shape and then having every config turned down at
    # generation would leave nothing behind, so the fallback is tried again
    # from here.  Reached only on that failure, so it costs the shapes a tier
    # serves nothing.
    if not offered and candidates[0][0] is not npu_triton_bmm_template:
        offered = offer([
            (npu_triton_bmm_template,
             _get_npu_bmm_configs(m, n, k, elem_bytes, batch, dtype))
        ])

    # Coming back empty is the outcome the fallback is supposed to make
    # impossible, and it is invisible from the outside: autotune simply picks
    # the operator library and reports a tie.  The count is checked rather
    # than any one of the paths that can produce it.
    if not offered:
        log.warning(
            "NPU Triton bmm: no candidate for bmm(%s, %s, %s) batch=%s "
            "dtype=%s -- aten.bmm will run from the operator library and no "
            "epilogue can fuse into it.",
            m, n, k, batch, mat1.get_dtype(),
        )


def is_batch_stride_largest_or_zero(mat1, mat2, layout) -> bool:
    """
    Checking if the batch stride is the largest in the stride.
    """
    sizes = [mat1.get_size(), mat2.get_size(), layout.size]
    strides = [mat1.get_stride(), mat2.get_stride(), layout.stride]
    for size, stride in zip(sizes, strides):
        assert len(size) == len(stride) == 3, "Expect 3D tensors"
        if stride[0] != 0 and stride[0] != sympy_product(size[1:]):
            return False

    return True


def _register_npu_inductor_bmm():
    @L.register_lowering(aten.bmm)
    def tuned_bmm(mat1, mat2, *, layout=None):
        if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
            # decompose to small ops when memory bound
            if mat1.get_size()[1] == 1 or mat2.get_size()[2] == 1:
                mat1 = L.unsqueeze(mat1, -1)
                mat2 = L.unsqueeze(mat2, 1)
                return L.sum_(L.mul(mat1, mat2), axis=2)

            def is_valid_to_require_contiguous(t):
                if not ir.is_storage_and_layout(t):
                    return True
                _, layout = ir.as_storage_and_layout(t, freeze=False)
                return isinstance(layout, ir.FlexibleLayout)

            def is_preferred_layout_as_bmm_input(sizes, strides):
                # contiguous on one of the last two dims
                return (
                    strides[-1] == 1 and (sizes[-2] == 1 or strides[-2] >= sizes[-1])
                ) or (strides[-2] == 1 and (sizes[-1] == 1 or strides[-1] >= sizes[-2]))

            # Make the input of bmm contiguous
            # if it is not contiguous on either of the last two dims,
            # because bmm cpu implementation would do contiguous() if not.
            # This is to avoid additional copies in bmm.
            def may_require_contiguous(t, meta_t):
                sizes = meta_t.meta["val"].size()
                strides = meta_t.meta["val"].stride()
                if not is_preferred_layout_as_bmm_input(sizes, strides):
                    t = ir.ExternKernel.require_contiguous(t)
                return t

            if is_valid_to_require_contiguous(mat1):
                meta_mat1 = V.graph.current_node.args[0]
                mat1 = may_require_contiguous(mat1, meta_mat1)
            if is_valid_to_require_contiguous(mat2):
                meta_mat2 = V.graph.current_node.args[1]
                mat2 = may_require_contiguous(mat2, meta_mat2)
        m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

        # options to tune from
        choices = (
            [aten_bmm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
        )
        static_shape, is_nonzero = _is_static_problem(layout)

        # The extents that leave the cube with nothing to reuse are answered by
        # decomposing instead of tiling.  Behind the same gate as the templates
        # below, because it stands in for them: with the Triton path switched
        # off the caller has asked for the operator library.
        if is_nonzero and use_triton_template(layout) and _degenerate_axis(
            mat1, mat2
        ):
            mat1 = L.unsqueeze(mat1, -1)
            mat2 = L.unsqueeze(mat2, 1)
            return L.sum_(L.mul(mat1, mat2), axis=2)

        batch_stride_largest_or_zero = is_batch_stride_largest_or_zero(mat1, mat2, layout)
        is_contiguous_input = False
        if batch_stride_largest_or_zero:
            is_contiguous_input = (
                is_contiguous_striding(mat1.get_size()[1:], mat1.get_stride()[1:])
                and is_contiguous_striding(mat2.get_size()[1:], mat2.get_stride()[1:])
            )
        if (
            is_contiguous_input
            and static_shape
            and is_nonzero
            and use_catlass_template("bmm", layout, m, n, k)
        ):
            from ..codegen.catlass.gemm_template import CATLASS1xGemmTemplate

            CATLASS1xGemmTemplate.add_catlass_gemm_choices(
                choices, layout, [mat1, mat2]
            )

        if use_cpp_bmm_template(layout, mat1, mat2):
            from torch._inductor.codegen.cpp_bmm_template import CppBmmTemplate

            CppBmmTemplate.add_choices(
                choices,
                layout,
                [mat1, mat2],
            )
        if use_ck_template(layout):
            CKGemmTemplate.add_ck_gemm_choices(choices, layout, [mat1, mat2])

        # Add NPU Triton bmm template choices for CV (Compute/Vector) fusion.
        # The bmm template carries the batch dimension in the linear tile index
        # and supports epilogue fusion via {{store_output}}.
        if is_nonzero and use_triton_template(layout):
            try:
                # T2 and T3 bake the batch count into their loops, so they
                # are only offered on static, contiguous shapes; 0 tells
                # add_npu_triton_bmm_choices to offer the generic one alone.
                batch = 0
                if static_shape and is_contiguous_input:
                    try:
                        batch = int(mat1.get_size()[0])
                    except TypeError:
                        batch = 0
                add_npu_triton_bmm_choices(
                    choices, layout, mat1, mat2, m, n, k, batch
                )
                log.debug(
                    "NPU Triton CV fusion: added triton bmm template choices "
                    "for bmm(%d, %d, %d), total choices now %d",
                    m,
                    n,
                    k,
                    len(choices),
                )
            except Exception:
                log.warning("Failed to add NPU triton bmm template choices", exc_info=True)

        if len(choices) == 0:
            log.warning("No choices for GEMM, using ATen backend as fallback")
            choices.append(aten_bmm.bind((mat1, mat2), layout))

        node, _ = autotune_select_algorithm(
            "bmm", choices, [mat1, mat2], layout
        )
        return node
