import logging
from typing import Any, Dict, List, Optional

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
from ..select_algorithm import NPUTritonTemplate, NPUTemplateCompileOption
from ..utils import use_catlass_template, use_triton_template
from torch_npu.npu import matmul
from .mm_common import (
    ACCUM_BYTES,
    dtype_to_bytes,
    L0C_BYTES,
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

# Operand element size assumed when the dtype is unknown.  Two bytes is the
# conservative choice: it is what every capacity check below was tuned at, and
# guessing smaller would admit plans that overrun L1 on the device.
DEFAULT_ELEM_BYTES = 2

# A+B working set above which L2 is bypassed rather than allocated.  Below it a
# second read of B is an L2 hit worth keeping; above it the operands evict each
# other and no-allocate is the better hint.
L2_BYPASS_FROM_BYTES = 80 * 1024 * 1024

# How many exact (non power-of-two) N widths to offer.  The candidate list is
# also the list autotune re-times with the epilogue fused, and that window is
# only a few entries deep, so it is kept short on purpose.
MAX_EXACT_N_WIDTHS = 5


# ---------------------------------------------------------------------------
# Generic BMM template: one program per (batch, M tile, N tile)
# ---------------------------------------------------------------------------
# The mm template plus a batch dimension (idx_q = tl.program_id(1)).  Always
# offered, so there is a Triton choice for any shape, and {{store_output}} keeps
# it open to epilogue fusion.

@SymbolicGridFn
def npu_bmm_grid(b, m, n, meta, *, cdiv):
    """Grid for the generic template: (M*N tiles, batch, 1)."""
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), b, 1)


def _split_groups(n_groups: int) -> tuple[int, int, bool]:
    """Split n_groups over programs, as (programs, per_program, ragged).

    A program has to own a *run* of groups: separate program instances cannot
    overlap, each re-paying kernel entry with its pipes drained, while a run
    becomes an scf.for where iteration i+1's MTE2 runs under iteration i's MAC.

    Exact divisors alone leave cores idle whenever the group count has no
    divisor near the core count, which a power-of-two batch never has.  So the
    best exact split is weighed against handing every core a run and letting the
    last ones come up short; a group is whole, so coming up short costs a trip
    count and no mask.
    """
    exact = next((p for p in range(min(n_groups, NUM_CUBE_CORES), 0, -1)
                  if n_groups % p == 0), 1)
    programs = min(n_groups, NUM_CUBE_CORES)
    per_program = -(-n_groups // programs)
    if n_groups // exact <= per_program:
        return exact, n_groups // exact, False
    return programs, per_program, True


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

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # A ragged M or N tail has to be masked at the load, not just at the store:
    # on the last batch the overhanging rows/columns run off the end of the
    # tensor and the MTE reports an out-of-range DDR address (aicore error 95).
    # Masking, rather than clamping the index, is what keeps the address affine
    # -- `tl.minimum` on rm makes the backend materialize the whole 2D index
    # tensor in UB and every wide tile then fails to fit.
    {% if not EVEN_M %}
    m_mask = rm < M
    {% endif %}
    {% if not EVEN_N %}
    n_mask = rn < N
    {% endif %}

    # batch dimension index — precompute batch offsets before K-loop
    # to avoid redundant multiply inside the hot loop
    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    a_batch_off = idx_q * stride_aq
    b_batch_off = idx_q * stride_bq

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        {% if not EVEN_K %}
        k_mask = offs_k < K
        {% endif %}
        a = tl.load(A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak + a_batch_off){% if not (EVEN_M and EVEN_K) %}, mask={% if not EVEN_M %}m_mask[:, None]{% if not EVEN_K %} & {% endif %}{% endif %}{% if not EVEN_K %}k_mask[None, :]{% endif %}, other=0.0{% endif %})
        b = tl.load(B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn + b_batch_off){% if not (EVEN_K and EVEN_N) %}, mask={% if not EVEN_K %}k_mask[:, None]{% if not EVEN_N %} & {% endif %}{% endif %}{% if not EVEN_N %}n_mask[None, :]{% endif %}, other=0.0{% endif %})
        acc = tl.dot(a, b, acc=acc{% if ALLOW_HF32 %}, input_precision="tf32"{% endif %}, out_dtype=ACC_TYPE)

    # rematerialize rm, rn and idx_q to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
"""

# Compile options rather than kernel constexprs. A key not listed here is
# emitted as `tl.constexpr` and never reaches bishengir-compile, which silently
# drops --enable-hivm-batch-matmul and lowers the rank-3 dot as a batch loop.
_BMM_COMPILE_OPTIONS = NPUTemplateCompileOption(
    {
        "enable_hivm_batch_matmul": True,
        "unit_flag": False,
    }
)

npu_triton_bmm_template = NPUTritonTemplate(
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
) -> List[Dict[str, Any]]:
    """Tilings for the generic template, as (BLOCK_M, BLOCK_N, BLOCK_K).

    The same tiling shapes mm offers, plus the wider ones autotune picked up on
    large BMM shapes: BLOCK_N=256 for wide outputs, and BLOCK_K=128/256 to cut
    the K-loop trip count on long reductions.
    """
    configs: List[Dict[str, Any]] = []

    tile_shapes = [
        (64, 64, 32),
        (64, 128, 32),
        (128, 64, 32),
        (128, 128, 32),
        (64, 64, 64),
        (64, 64, 128),
        (64, 64, 256),
        (64, 256, 256),
        (128, 64, 64),
        (64, 128, 64),
        (128, 128, 64),
        (128, 256, 64),
        (64, 256, 64),
        (128, 128, 128),
        (64, 128, 128),
        (128, 256, 256),
        (128, 128, 256),
        (32, 64, 32),
        (64, 32, 32),
        (32, 32, 32),
        (32, 32, 128)
    ]

    # A BLOCK_K that does not divide K masks every K tile, costing a fill per
    # tile for no extra coverage.  It can still tie on the bare bmm, which is
    # where autotune measures, and then lose once the epilogue is fused: at
    # B32/M200/N200/K1600 the two widths gave 1.10x and 0.92x fused.  Masked
    # widths are therefore offered only when no width divides K, as at K=200.
    aligned = [shape for shape in tile_shapes if k % shape[2] == 0]
    for block_m, block_n, block_k in (aligned or tile_shapes):
        even_k = (k % block_k == 0)
        # GROUP_M: how many M tiles are grouped before advancing N, which is the
        # traversal order and so the L2 reuse.  Row-major (1) is offered only on
        # large tiles, where grid_m is small enough for grouping to buy nothing.
        if block_m >= 128 and block_n >= 128:
            group_m_values = [1, 8]
        else:
            group_m_values = [8]
        # One pipeline depth rather than a 2/3 sweep: the two depths of a tiling
        # land within noise of each other, so offering both spends the fused
        # re-measure window on near-duplicates instead of on different plans.
        for group_m in group_m_values:
            configs.append({
                # The tile one program owns, and the K step it reduces over.
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
                # Depth the frontend runs the K loop ahead of itself, and the
                # warp count, which the cube path does not vary.
                "num_stages": 2,
                "num_warps": 4,
                # Whether the mmad may take the tf32 path, and the accumulator
                # type, which stays fp32 whatever the operands are.
                "ALLOW_HF32": matmul.allow_hf32,
                "ACC_TYPE": "tl.float32",
                # Whether each axis divides its tile.  An axis that does not is
                # masked at the load, not just at the store.
                "EVEN_K": even_k,
                "EVEN_M": (m % block_m == 0),
                "EVEN_N": (n % block_n == 0),
            })

    return configs


# ---------------------------------------------------------------------------
# Persistent BMM template: one program owns a run of whole matmuls
# ---------------------------------------------------------------------------
# The generic template re-reads both operands from GM for every K panel, which
# costs in both directions.  Small per-batch working sets are bound by
# per-block overhead rather than bandwidth, and large single matmuls are bound
# on MTE2, where re-reading A once per N tile adds about half again to the
# compulsory traffic.
#
# One kernel covers both.  A task is BLOCK_Q consecutive batches by all N tiles,
# with A's K panels hoisted out of the N loop: BLOCK_Q=1 over several N tiles is
# the A-resident schedule, BLOCK_Q>1 over a single N tile is the grouped one,
# and a persistent loop over the cube cores amortises kernel entry in both.
# The tiling is derived from the L0/L1 capacities rather than from a shape
# table, so a shape outside the tuned set still gets an answer.
#
# Note compact_l0c_m is deliberately unused: trimming the L0C tile below
# BLOCK_M makes the mmad's M differ from it, which is the condition that sends
# the unit flag down its faulting path.


def _next_pow2(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def _pow2_parts(value: int) -> List[int]:
    """value as descending powers of two, which is what tl.arange can express."""
    parts = []
    bit = 1 << value.bit_length()
    while value:
        bit >>= 1
        if value & bit:
            parts.append(bit)
            value -= bit
    return parts


def _fractal_rows(extent: int) -> int:
    """``extent`` as the hardware allocates it: whole 16-row fractals.

    BLOCK_M=200 occupies 13 of them, so a buffer budgeted on the raw 200 is
    eight rows short of what L0C and L1 hand out.
    """
    return -(-extent // MMAD_M_FRACTAL) * MMAD_M_FRACTAL


def _panel_offsets(panels: List[int]) -> List[int]:
    """Where each panel starts, as the running sum of the widths before it."""
    offsets, used = [], 0
    for width in panels:
        offsets.append(used)
        used += width
    return offsets


def _k_panels(k: int, block_k: int) -> List[int]:
    """How K is cut into mmad panels, in the order the kernel loads them.

    K is peeled rather than padded, since a panel padded past K reads as inf,
    and every panel is one tl.arange, so every width is a power of two: the
    remainder BLOCK_K leaves becomes one panel per set bit.

    Splitting K is free of rounding only while every panel boundary is a whole
    number of MMAD fractals, so the remainder leads only when it is itself a
    whole number of them and otherwise trails.  Leading is preferred where
    available, the prologue having no mmad to overlap its wait away.

    A K that is not a whole number of fractals cannot be covered exactly, so
    such a tail is rounded up and the caller masks the overhang, which loads as
    zero and drops out of the dot.  Peeling it to a narrower panel is not an
    option: the mmad reads a fractal whole, so a partial one is uninitialised L1
    and PlanMemory rejects the kernel outright.
    """
    full, rem = divmod(k, block_k)
    if not rem:
        return [block_k] * full
    if rem % MMAD_K_FRACTAL == 0:
        return _pow2_parts(rem)[::-1] + [block_k] * full
    tail = -(-rem // MMAD_K_FRACTAL) * MMAD_K_FRACTAL
    return [block_k] * full + _pow2_parts(tail)


def _m_candidates(m: int) -> List[int]:
    """BLOCK_M to search: the rounded-up tile, two halvings of it, and M itself.

    Two halvings is the depth a large M needs to reach a tile whose A still fits
    L1 at a long K.  M itself is offered because tl.arange needs a power of two
    only in SIMT mode, which this template does not compile under, and rounding
    to one is sometimes exactly what pushes A out of L1.
    """
    whole = _next_pow2(m)
    # 32 is admitted for the batched shapes, where M=48 padded to 64 is a third
    # of every mmad; 16 is not, being a width no large shape wants.
    cands = [bm for bm in (whole, whole // 2, whole // 4) if bm >= 32]
    if m >= 32 and m not in cands:
        cands.append(m)
    return cands


@SymbolicGridFn
def npu_bmm_persistent_grid(b, m, n, meta, *, cdiv):
    """Grid for the persistent template: one program per cube core."""
    return (meta["CORES"], 1, 1)


# `extension` is not imported here: torch_npu patches gen_common_triton_imports
# to bind it in the generated module, and def_kernel puts this body inside the
# kernel function, where an import line could not go anyway.
_BMM_PERSISTENT_TEMPLATE = r"""{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    core = tl.program_id(0)

    offs_q = tl.arange(0, BLOCK_Q)
    offs_n0 = tl.arange(0, BLOCK_N)

    # Rank 3 throughout, so tl.dot emits BatchMmadL1Op.  At BLOCK_Q=1 the
    # leading 1 makes BatchL1Mmad delegate to L1Mmad, the path that gets the L0
    # K64 ping-pong; rank 2 reaches the same mmad but not the same scheduling.

    # The task count need not divide the core count, and the tail is guarded
    # rather than clamped: clamping would make tail cores repeat a task, and a
    # repeated {{store_output}} is not idempotent once an epilogue is fused.
    for task_idx in range(TASKS_PER_CORE):
{% if CONTIGUOUS_TASKS %}
        # Contiguous rather than strided by the core count, so one program walks
        # one slab of A and B instead of revisiting the whole tensor.
        task = core * TASKS_PER_CORE + task_idx
{% else %}
        task = core + task_idx * CORES
{% endif %}
        if task < TOTAL_TASKS:
{% if N_GROUPS > 1 %}
            # N group is the minor index.  A batch's groups differ only in which
            # columns of B they take, so putting them on neighbouring cores lets
            # the A they all re-read be found in L2 by everyone after the first.
            n_group = task % N_GROUPS
            slot = task // N_GROUPS
{% else %}
            slot = task
{% endif %}
{% if M_TILES > 1 %}
            # M tile is the minor index, so tiles sharing a B column land on
            # neighbouring cores and the second reader finds it still in L2.
            # Making it the major index instead measured 16% slower, which is
            # what that hit is worth.
            q = (slot // M_TILES) * BLOCK_Q + offs_q
{% if SHIFT_M %}
            # M does not split into whole tiles, so the last tile is pulled back
            # to end at M rather than masked: a mask on the outer axis of the
            # [Q, M, N] tile stops the drain being one descriptor, which costs
            # 4x once an epilogue is fused.  The pulled-back tile recomputes the
            # rows it now shares with its predecessor and stores the same values
            # into them, which an elementwise epilogue is indifferent to.
            m_base = tl.minimum((slot % M_TILES) * BLOCK_M, M - BLOCK_M)
{% else %}
            m_base = (slot % M_TILES) * BLOCK_M
{% endif %}
{% else %}
            q = slot * BLOCK_Q + offs_q
            m_base = 0
{% endif %}
            idx_q = q[:, None, None]
            b_base = B + idx_q * stride_bq
            # The tile's rows are covered by M_PANELS, one mmad each.  Panels
            # share the B tile they are interleaved with, so covering M this way
            # costs no extra read of B.
{% for p in range(M_PANELS | length) %}
            offs_m_{{ p }} = m_base + {{ M_PANEL_OFFSETS[p] }} + tl.arange(0, {{ M_PANELS[p] }})
            a_base_{{ p }} = A + idx_q * stride_aq + offs_m_{{ p }}[None, :, None] * stride_am
{% if MASKED_M %}
            m_mask_{{ p }} = offs_m_{{ p }}[None, :, None] < M
{% endif %}
{% if STREAMED and MASKED_M %}
            # Streamed panels read through a clamped row index rather than a row
            # mask: a masked A load re-issued per N tile faults the device
            # (507015).  Rows past M reread row M-1 and are never stored, since
            # the epilogue store is masked on the same bound.
            a_stream_base_{{ p }} = (A + idx_q * stride_aq
                             + tl.minimum(offs_m_{{ p }}, M - 1)[None, :, None] * stride_am)
{% endif %}
{% endfor %}

{% if not PIPELINED %}
            # A's resident K panels, loaded once ahead of the N loop.
            # static_range so the frontend unrolls it and the N loop can index
            # a_tiles with a trace-time constant; concatenation rather than
            # .append, since the frontend models a Python list as a tl.tuple.
            #
            # These temporaries are named apart from the N loop's: panels are
            # not all one width, so a name assigned in both places becomes a
            # loop-carried variable whose type changes and the frontend rejects
            # the kernel.
{% for p in range(M_PANELS | length) %}
            a_tiles_{{ p }} = ()
            for ki in tl.static_range(0, RESIDENT_K_TILES):
                offs_kr = K_OFFSETS[ki] + tl.arange(0, K_PANELS[ki])
                ar_ptrs_{{ p }} = a_base_{{ p }} + offs_kr[None, None, :] * stride_ak
{% if K_MASKED %}
                # ki is a static_range index, so the arm this plan does not use
                # is gone before codegen.
                if ki < K_MASK_FROM:
                    ar_{{ p }} = tl.load(ar_ptrs_{{ p }}{% if MASKED_M %}, mask=m_mask_{{ p }}, other=0.0{% endif %})
                else:
                    ar_{{ p }} = tl.load(ar_ptrs_{{ p }}, mask={% if MASKED_M %}m_mask_{{ p }} & {% endif %}(offs_kr[None, None, :] < K), other=0.0)
{% elif MASKED_M %}
                ar_{{ p }} = tl.load(ar_ptrs_{{ p }}, mask=m_mask_{{ p }}, other=0.0)
{% else %}
                ar_{{ p }} = tl.load(ar_ptrs_{{ p }})
{% endif %}
{% if L1_COPIES %}
                extension.multibuffer(ar_{{ p }}, L1_COPIES)
{% endif %}
{% if L2_MODE_A %}
                extension.compile_hint(ar_{{ p }}, "l2_cache_mode", L2_MODE_A)
{% endif %}
                a_tiles_{{ p }} = a_tiles_{{ p }} + (ar_{{ p }},)
{% endfor %}
{% endif %}

            # No guard on this loop: N_TILES is exact by construction, and the
            # ragged last tile is handled by MASKED_N rather than by a bound.
            for n_tile in range(N_TILES):
                offs_n = ({% if N_GROUPS > 1 %}n_group * N_TILES + {% endif %}n_tile) * BLOCK_N + offs_n0
{% if MASKED_N %}
                n_mask = offs_n[None, None, :] < N
{% endif %}
{% for p in range(M_PANELS | length) %}
                acc_{{ p }} = tl.zeros((BLOCK_Q, {{ M_PANELS[p] }}, BLOCK_N), dtype=ACC_TYPE)
{% endfor %}

{% if PIPELINED %}
                # Each panel is loaded one iteration ahead of the mmad that
                # consumes it.  Loading and using a panel in the same iteration
                # -- the else arm -- leaves their lifetimes disjoint, so
                # PlanMemory puts every panel on one L1 slot and no load can run
                # under the previous mmad.  Two live panels force two slots.
                offs_kp = K_OFFSETS[0] + tl.arange(0, K_PANELS[0])
                b_next = tl.load(
                    b_base
                    + offs_kp[None, :, None] * stride_bk
                    + offs_n[None, None, :] * stride_bn
{%- if MASKED_N %}, mask=n_mask, other=0.0{% endif %})
                extension.compile_hint(b_next, "dot_pad_only_k")
{% if L2_MODE_B %}
                extension.compile_hint(b_next, "l2_cache_mode", L2_MODE_B)
{% endif %}
{% for p in range(M_PANELS | length) %}
                a_next_{{ p }} = tl.load(
{%- if MASKED_M %}a_stream_base_{{ p }}{% else %}a_base_{{ p }}{% endif %}
                    + offs_kp[None, None, :] * stride_ak)
{% endfor %}

                for ki in tl.static_range(0, K_TILES):
                    b_cur = b_next
{% for p in range(M_PANELS | length) %}
                    a_cur_{{ p }} = a_next_{{ p }}
{% endfor %}
                    # ki is static, so the last arm disappears at codegen.
                    if ki + 1 < K_TILES:
                        offs_kn = K_OFFSETS[ki + 1] + tl.arange(0, K_PANELS[ki + 1])
                        b_next = tl.load(
                            b_base
                            + offs_kn[None, :, None] * stride_bk
                            + offs_n[None, None, :] * stride_bn
{%- if MASKED_N %}, mask=n_mask, other=0.0{% endif %})
                        extension.compile_hint(b_next, "dot_pad_only_k")
{% if L2_MODE_B %}
                        extension.compile_hint(b_next, "l2_cache_mode", L2_MODE_B)
{% endif %}
{% for p in range(M_PANELS | length) %}
                        a_next_{{ p }} = tl.load(
{%- if MASKED_M %}a_stream_base_{{ p }}{% else %}a_base_{{ p }}{% endif %}
                            + offs_kn[None, None, :] * stride_ak)
{% endfor %}
{% for p in range(M_PANELS | length) %}
                    acc_{{ p }} = tl.dot(
                        a_cur_{{ p }},
                        b_cur,
                        acc=acc_{{ p }},
{% if ALLOW_HF32 %}
                        input_precision="tf32",
{% endif %}
                        out_dtype=ACC_TYPE,
                    )
{% endfor %}
{% else %}
                # The B loads are interleaved with the mmads rather than hoisted
                # above them, so each tile's lifetime ends at its mmad and
                # PlanMemory can coalesce them onto one L1 slot.  Hoisting them
                # all was measured and buys nothing.
                for ki in tl.static_range(0, K_TILES):
                    offs_k = K_OFFSETS[ki] + tl.arange(0, K_PANELS[ki])
                    b_ptrs = (
                        b_base
                        + offs_k[None, :, None] * stride_bk
                        + offs_n[None, None, :] * stride_bn
                    )
{% if K_MASKED %}
                    if ki < K_MASK_FROM:
                        b = tl.load(b_ptrs{% if MASKED_N %}, mask=n_mask, other=0.0{% endif %})
                    else:
                        b = tl.load(b_ptrs, mask={% if MASKED_N %}n_mask & {% endif %}(offs_k[None, :, None] < K), other=0.0)
{% elif MASKED_N %}
                    b = tl.load(b_ptrs, mask=n_mask, other=0.0)
{% else %}
                    b = tl.load(b_ptrs)
{% endif %}
{% if L1_COPIES %}
                    extension.multibuffer(b, L1_COPIES)
{% endif %}
                    # Without this the compiler takes the mmad's N from the
                    # padded fractal shape and computes the ragged tail at full
                    # BLOCK_N.
                    extension.compile_hint(b, "dot_pad_only_k")
{% if L2_MODE_B %}
                    extension.compile_hint(b, "l2_cache_mode", L2_MODE_B)
{% endif %}
                    # A panel that did not fit comes back here, once per N tile,
                    # and carries no l2_cache_mode: unlike the resident panels
                    # it has N_TILES-fold reuse and wants to be cached.  ki is a
                    # static_range index, so the else arm disappears entirely on
                    # plans that hoist everything.  The panels are consumed here
                    # inside the K loop, while the B tile just staged is still
                    # live; draining them after the loop would re-read B once
                    # per panel.
{% for p in range(M_PANELS | length) %}
                    if ki < RESIDENT_K_TILES:
                        a_{{ p }} = a_tiles_{{ p }}[ki]
                    else:
{% if MASKED_M %}
                        a_ptrs_{{ p }} = a_stream_base_{{ p }} + offs_k[None, None, :] * stride_ak
{% else %}
                        a_ptrs_{{ p }} = a_base_{{ p }} + offs_k[None, None, :] * stride_ak
{% endif %}
{% if K_MASKED %}
                        if ki < K_MASK_FROM:
                            a_{{ p }} = tl.load(a_ptrs_{{ p }})
                        else:
                            a_{{ p }} = tl.load(a_ptrs_{{ p }}, mask=offs_k[None, None, :] < K, other=0.0)
{% else %}
                        a_{{ p }} = tl.load(a_ptrs_{{ p }})
{% endif %}
                    acc_{{ p }} = tl.dot(
                        a_{{ p }},
                        b,
                        acc=acc_{{ p }},
{% if ALLOW_HF32 %}
                        input_precision="tf32",
{% endif %}
                        out_dtype=ACC_TYPE,
                    )
{% endfor %}
{% endif %}

                idx_n = offs_n[None, None, :]
{% for p in range(M_PANELS | length) %}
                idx_m_{{ p }} = offs_m_{{ p }}[None, :, None]
{% if MASKED_M and MASKED_N %}
                mask_{{ p }} = m_mask_{{ p }} & n_mask
{% elif MASKED_M %}
                mask_{{ p }} = m_mask_{{ p }}
{% elif MASKED_N %}
                mask_{{ p }} = n_mask
{% else %}
                mask_{{ p }} = None
{% endif %}
                {{store_output(("idx_q", "idx_m_" ~ p, "idx_n"), "acc_" ~ p, "mask_" ~ p, val_shape=("BLOCK_Q", M_PANELS[p], "BLOCK_N"), indent_width=16)}}
{% endfor %}
"""

npu_triton_bmm_persistent_template = NPUTritonTemplate(
    name="npu_triton_bmm_persistent",
    grid=npu_bmm_persistent_grid,
    source=_BMM_PERSISTENT_TEMPLATE,
    debug=False,
    compile_options=_BMM_COMPILE_OPTIONS,
)


# ---------------------------------------------------------------------------
# Batched BMM template: BLOCK_Q batches per rank-3 mmad
# ---------------------------------------------------------------------------
# The grouped regime: small per-batch working sets where a whole matmul fits in
# one tile and many batches fit in L1 at once.  One program walks
# GROUPS_PER_PROGRAM consecutive groups, each a rank-3 tl.dot over BLOCK_Q
# batches, which lowers to BatchMmadL1Op.  The group loop is an scf.for rather
# than separate program instances so iteration i+1's MTE2 runs under iteration
# i's MAC, and extension.multibuffer marks the L1 staging buffers so the
# compiler double-buffers across groups.
#
# Unlike the persistent template there is no A-residency (a single N tile makes
# hoisting A pointless) and no K peeling (BLOCK_K == K in this regime).

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

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_q = tl.arange(0, BLOCK_Q)

    # Contiguous rather than strided by the program count, so one program walks
    # one slab of A and B instead of revisiting the whole tensor
    # GROUPS_PER_PROGRAM times.
    first_q = tl.program_id(0) * (GROUPS_PER_PROGRAM * BLOCK_Q)

    # Every group is one nd2nz -> nd2nz -> mmad -> fixpipe chain with no slack
    # in it; only a neighbouring group can fill the idle pipes, and only if the
    # compiler can see it, which means it has to be a loop body.
{% if RAGGED_GROUPS %}
    # The group count has no divisor near the core count, which a power-of-two
    # batch never has, so every core gets a run and the last ones come up short.
    # A group is whole, so this is a trip count and nothing else, with no element
    # mask anywhere in the body.  Clamping the overrun onto the last group to
    # keep the count static measured the same and writes the same addresses from
    # two cores at once, so the shorter run is preferred.
    n_groups = tl.maximum(0, tl.minimum(GROUPS_PER_PROGRAM, (BATCH - first_q) // BLOCK_Q))
{% else %}
    n_groups = GROUPS_PER_PROGRAM
{% endif %}
    for g in range(n_groups):
        q = first_q + g * BLOCK_Q + offs_q
        idx_q = q[:, None, None]

        a_ptrs = A + idx_q * stride_aq + offs_m[None, :, None] * stride_am + offs_k[None, None, :] * stride_ak
{% if MASKED_M or MASKED_K %}
        a = tl.load(a_ptrs, mask=(offs_m[None, :, None] < M) & (offs_k[None, None, :] < K), other=0.0)
{% else %}
        a = tl.load(a_ptrs)
{% endif %}
{% if L1_COPIES %}
        extension.multibuffer(a, L1_COPIES)
{% endif %}

        b_ptrs = B + idx_q * stride_bq + offs_k[None, :, None] * stride_bk + offs_n[None, None, :] * stride_bn
{% if MASKED_K or MASKED_N %}
        b = tl.load(b_ptrs, mask=(offs_k[None, :, None] < K) & (offs_n[None, None, :] < N), other=0.0)
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
        {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_Q", "BLOCK_M", "BLOCK_N"), indent_width=8)}}
"""

npu_triton_bmm_batched_template = NPUTritonTemplate(
    name="npu_triton_bmm_batched",
    grid=npu_bmm_batched_grid,
    source=_BMM_BATCHED_TEMPLATE,
    debug=False,
    compile_options=_BMM_COMPILE_OPTIONS,
)


def _wave_use(tasks: int) -> float:
    """Share of the core-cycles a task count spends on tasks rather than idle."""
    waves = -(-tasks // NUM_CUBE_CORES)
    return tasks / (waves * NUM_CUBE_CORES)


def _n_group_candidate(n_tiles: int, base_tasks: int) -> int:
    """The N split worth offering beside the undivided one, or 1 for none.

    Splitting N is the only lever left on the task count once M is whole, and
    it is wanted for two different reasons at once: it refills the last wave,
    and it shortens the run of N tiles a task walks, which is what decides how
    often the K panels that did not fit L1 are read again.  Only the best
    divisor is offered -- the candidate list is also the list autotune re-times
    with the epilogue fused, and that window is four entries deep.
    """
    best, best_use = 1, _wave_use(base_tasks)
    for groups in range(2, n_tiles + 1):
        if n_tiles % groups:
            continue
        use = _wave_use(base_tasks * groups)
        if use > best_use + 1e-9:
            best, best_use = groups, use
    return best


def _persistent_config(
    batch: int,
    m: int,
    n: int,
    k: int,
    block_q: int,
    block_m: int,
    block_n: int,
    block_k: int,
    l2_mode_a: int,
    l2_mode_b: int,
    contiguous: bool,
    l1_copies: int,
    elem_bytes: int = DEFAULT_ELEM_BYTES,
    shave: int = 0,
    n_groups: int = 1,
    stages: int = 4,
    pipelined: bool = False,
) -> Optional[Dict[str, Any]]:
    """One config for the persistent template, or None if it does not fit.

    Everything here is a capacity check rather than a shape test, so a shape
    outside the ones this was tuned on gets an answer rather than a default.
    ``elem_bytes`` is threaded in rather than assumed, since at four bytes every
    L1 budget below doubles.
    """
    # mm_args hands back sympy Integers, which carry enough of int's protocol
    # to reach _k_panels before bit_length fails.  The caller catches and logs
    # that raise, which becomes a silent fall back to ATen, so coerce here.
    batch, m, n, k = int(batch), int(m), int(n), int(k)

    # The task computes its rows as a single mmad, so there is exactly one M
    # panel and every budget below is taken on the tile height.
    panels = [block_m]
    m_panel_offsets = _panel_offsets(panels)

    if batch % block_q:
        # A masked batch costs a full linalg.fill on MTE2.
        return None

    if k < block_k:
        # Below one BLOCK_K the panels are K's binary decomposition rather than
        # a peeled tail, and those plans fault the device (507015) at K=80.
        return None

    widths = _k_panels(k, block_k)
    offsets = _panel_offsets(widths)
    k_tiles = len(widths)

    # Panels over-cover a K that is not a whole number of fractals, so the
    # loads from here on read past K and have to be masked.
    k_mask_from = next((i for i, (off, width) in enumerate(zip(offsets, widths))
                        if off + width > k), k_tiles)
    n_tiles = (n + block_n - 1) // block_n
    # A task owns a slice of N rather than all of it.  Both operands are read
    # once per task, so an M split re-reads B while an N split re-reads A, and B
    # is the larger of the two by however much N exceeds M.  Splitting N also
    # gives back the task count that keeping M whole costs.
    if n_tiles % n_groups:
        return None
    n_tiles_per_group = n_tiles // n_groups
    m_tiles = (m + block_m - 1) // block_m

    # B's slot is one tile, doubled because MarkMultiBuffer rings the ND2NZ
    # staging buffer.  Whatever L1 is left after it holds A.
    b_ring = max(2, l1_copies) * block_q * block_k * block_n * elem_bytes
    if b_ring >= L1_BYTES:
        return None

    # A's panels are hoisted above the N loop and all live at once, so full
    # residency needs BLOCK_M * K * elem_bytes of what L1 has left.  A long K
    # stops fitting, so residency is a prefix: the panels that fit are hoisted
    # and the rest are re-read inside the N loop.  A gives way rather than B,
    # being the smaller operand and the one with cross-tile reuse.
    #
    # A streamed panel needs its own staging slot next to B's ring, taken from
    # the same budget, so residency gives up two panels the first time it stops
    # being whole.
    row_bytes = block_q * _fractal_rows(block_m) * elem_bytes

    def _resident_panels(budget: int) -> int:
        used = count = 0
        for width in widths:
            if used + width * row_bytes > budget:
                break
            used += width * row_bytes
            count += 1
        return count

    # One slot for the streamed panels, and it is a ceiling rather than a
    # choice: PlanMemory coalesces their disjoint lifetimes, so panel k+1 cannot
    # load under panel k's mmad and the streamed half of the K loop gets no
    # MTE2/MAC overlap.  Asking for a second slot with extension.multibuffer
    # faults the device (507015) even with the bytes reserved.
    resident = _resident_panels(L1_BYTES - b_ring)
    if resident < k_tiles:
        resident = _resident_panels(L1_BYTES - b_ring - block_k * row_bytes)
    if resident < 1:
        return None
    if shave:
        # Hoisting the last panel is not always better than streaming it: it
        # only lengthens a prologue that has no mmad to overlap, while the
        # re-reads may still be L2 hits.  Which side wins turns on whether that
        # panel fits L2 beside the B ring, so both are offered and measured.
        if resident < k_tiles or k_tiles < 3:
            return None
        resident -= shave
    if (n_tiles_per_group > 1 and m_tiles > 1
            and sum(widths[:resident]) * 2 < k):
        # Under half of K resident this is no longer the A-resident schedule,
        # and such plans win autotune on the bare bmm then lose once the
        # epilogue is fused.  The test is conditional because both exits make
        # residency matter less: a task holding a single N tile re-reads
        # nothing, and one holding the whole of M reads B once, so what it gives
        # up by streaming A is re-reads of the smaller operand.
        return None
    if k_mask_from < k_tiles and resident < k_tiles:
        # The overhanging panel is the last one, so anything short of full
        # residency streams it and re-issues its masked load once per N tile.
        # That is the shape of load that faults the device at 507015, so leave
        # the shape to the plans that hoist the tail.
        return None
    # One L0C tile, at BLOCK_M rounded up to a whole fractal, which is the
    # height the hardware gives it either way.
    if block_q * _fractal_rows(block_m) * block_n * ACCUM_BYTES > L0C_BYTES:
        return None

    if pipelined:
        # Only where residency has nothing left to give: a plan that holds A
        # whole already reads every panel once.
        if resident >= k_tiles or shave:
            return None
        if k % block_k:
            # Unequal panel widths make the staged tile a loop-carried value
            # whose type changes from one iteration to the next, and the
            # frontend rejects that.
            return None
        # Three slots budgeted for the two that are live.  Plans that come out
        # level with L1 fail PlanMemoryRegBase, and one failing candidate takes
        # every candidate for the shape down with it.
        a_slot = block_q * block_k * _fractal_rows(block_m) * elem_bytes
        if 3 * a_slot + b_ring > L1_BYTES:
            return None
        resident = 0

    total_tasks = (batch // block_q) * m_tiles * n_groups
    cores = min(total_tasks, NUM_CUBE_CORES)
    return {
        # The tile one task owns: BLOCK_Q batches by BLOCK_M rows by BLOCK_N
        # columns, reduced BLOCK_K at a time.
        "BLOCK_Q": block_q,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        # How K is cut into mmad panels: the widths, where each starts, and from
        # which one the loads overhang K and need masking.
        "K_TILES": k_tiles,
        "K_PANELS": tuple(widths),
        "K_OFFSETS": tuple(offsets),
        "K_MASKED": k_mask_from < k_tiles,
        "K_MASK_FROM": k_mask_from,
        # How many of A's panels are hoisted above the N loop, and so held in L1
        # across every N tile; STREAMED says the rest are re-read inside it.
        "RESIDENT_K_TILES": resident,
        "STREAMED": resident < k_tiles,
        # Stage each panel one K iteration ahead of its mmad.  Only for plans
        # that stream A, where there is no residency left to win instead.
        "PIPELINED": pipelined,
        # How the output is divided: N tiles walked inside one task, N groups
        # spread across tasks, and M tiles across tasks.
        "N_TILES": n_tiles_per_group,
        "N_GROUPS": n_groups,
        "M_TILES": m_tiles,
        # The rows of one M tile, as the panels one mmad each covers.
        "M_PANELS": tuple(panels),
        "M_PANEL_OFFSETS": tuple(m_panel_offsets),
        # A ragged M is handled by shifting the last tile back onto M; only a
        # shape too short to hold one whole tile still needs the mask.
        "SHIFT_M": m % block_m != 0 and m >= block_m,
        "MASKED_M": m % block_m != 0 and m < block_m,
        "MASKED_N": n % block_n != 0,
        # The persistent loop: how many tasks there are, how many one core
        # walks, and how many cores are launched.
        "TOTAL_TASKS": total_tasks,
        "TASKS_PER_CORE": (total_tasks + cores - 1) // cores,
        "CORES": cores,
        # Whether a core's tasks are one contiguous run rather than strided by
        # the core count, so it walks one slab of A and B.
        "CONTIGUOUS_TASKS": contiguous,
        # Copies of the L1 staging buffer, which is what lets a load run under
        # the previous mmad.
        "L1_COPIES": l1_copies,
        # Per-operand L2 hint: 0 allocates, 2 no-allocate, 4 full bypass.
        # Bypass is right when the operands overrun L2 and wrong when they fit.
        "L2_MODE_A": l2_mode_a,
        "L2_MODE_B": l2_mode_b,
        # The same option that gates the batched macro also gates whether an
        # l2_cache_mode annotation is honoured, so the two modes above are inert
        # without it.
        "enable_hivm_batch_matmul": True,
        # Explicitly off, not merely unset: --enable-hivm-unit-flag-sync
        # defaults to true here, and the flag's disable state is device-global
        # rather than keyed by kernel, so any two kernels in one process alias
        # on flag group 0 and one's Fixpipe can make the next one's mmad skip a
        # FIX->M pair it needed.  Autotune puts a dozen variants of one shape in
        # a process, so this is reachable.  Leaving it on was measured and loses
        # anyway: the wait it removes is on memory, not on that handshake.
        "unit_flag": False,
        # How far ahead the frontend runs the loops.  Four was swept against 2,
        # 6 and 8; what the wide tilings lose is overlap on their own
        # dependences, which running further ahead does not recover.
        "num_stages": stages,
        "num_warps": 4,
        # Whether the mmad may take the tf32 path, and the accumulator type,
        # which stays fp32 whatever the operands are.
        "ALLOW_HF32": matmul.allow_hf32,
        "ACC_TYPE": "tl.float32",
    }


def _get_npu_bmm_persistent_configs(
    batch: int,
    m: int,
    n: int,
    k: int,
    elem_bytes: int = DEFAULT_ELEM_BYTES,
) -> List[Dict[str, Any]]:
    """Configs for the persistent template: the tiled, A-resident regime.

    One core owns a whole matmul, keeping A's K panels resident in L1 across
    every N tile, which suits large single matmuls.  The sweep below is over the
    tile shape, the L2 hints and how the output is divided; every candidate goes
    through ``_persistent_config``, which drops the ones that do not fit.
    """
    configs: List[Dict[str, Any]] = []
    seen = set()

    def add(block_q, block_m, block_n, block_k, l2_mode_a, l2_mode_b,
            contiguous, l1_copies, shave=0, n_groups=1,
            stages=4, pipelined=False):
        cfg = _persistent_config(
            batch, m, n, k, block_q, block_m, block_n, block_k,
            l2_mode_a, l2_mode_b, contiguous, l1_copies, elem_bytes, shave,
            n_groups, stages, pipelined,
        )
        if cfg is None:
            return
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key in seen:
            return
        seen.add(key)
        configs.append(cfg)

    # The N widths to sweep.  Both the mmad and B's read are paid on the covered
    # width, ceil(N / BLOCK_N) * BLOCK_N, rather than on N, so a width that
    # overhangs less can be worth a narrower B panel -- but only when it covers
    # N in appreciably fewer columns, since it also doubles the descriptor
    # count.  64 is therefore admitted on that test alone.
    n_cap = _next_pow2(n)
    n_widths = [128, 256, 512]
    if n_cap > 64 and -(-n // 64) * 64 * 5 <= -(-n // 128) * 128 * 4:
        n_widths.insert(0, 64)
    # Capped at the rounded-up N, and never left empty: a small N that is under
    # every width still needs one, or the template offers no config at all.
    n_widths = [w for w in n_widths if w <= n_cap] or [n_cap]

    # Widths that cover N in whole tiles, offered beside the powers of two.  A
    # tile is a tl.arange, which need not be a power of two (see
    # _m_candidates), and the power-of-two widths overhang: every overhanging
    # column is a B read and an mmad column outside the matrix.  Narrower than
    # these is left to the powers of two, a narrow B panel costing more MTE2
    # efficiency than the overhang does.
    exact = [n] + [n // t for t in range(2, 9)
                   if n % t == 0 and (n // t) % 16 == 0]
    added = 0
    for width in exact:
        if added == MAX_EXACT_N_WIDTHS:
            break
        if width <= 512 and width not in n_widths:
            n_widths.append(width)
            added += 1

    # And the widths that leave under one 16-column block of slop, which is what
    # an N that no width divides needs to be taken in a single tile.
    for n_tiles_exact in (1, 2, 3, 4):
        per = -(-n // n_tiles_exact)
        width = -(-per // 16) * 16
        if not 64 <= width <= min(-(-n // 16) * 16, 512) or width in n_widths:
            continue
        if width * n_tiles_exact - n < 16:
            n_widths.append(width)

    # M is tiled whole as well as split.  An undivided tile rounds up to a power
    # of two, which both wastes part of every mmad and is what stops a long K
    # fitting L1 at all; splitting is not free either, since every M tile walks
    # the whole N range and so re-reads B.  Both are offered and measured.
    for block_m in _m_candidates(m):
      for block_n in n_widths:
        # N_TILES=1 is allowed -- the persistent loop and K peeling still apply
        # without cross-tile reuse -- but a width past the rounded-up N only
        # pads B and C for no extra mmad width.
        if block_n > _next_pow2(n):
            continue
        # A width whose last tile is less than half used is padding that the
        # bare ranking does not charge for, a masked column being free without
        # an epilogue, and the plan then loses once the epilogue is on it.
        # Shapes covered in one or two tiles have no wider choice.
        if -(-n // block_n) >= 3 and n % block_n and 2 * (n % block_n) < block_n:
            continue
        for block_k in (64, 128, 256):
            if block_k > _next_pow2(k):
                continue
            # Which L2 pair to offer turns on whether the operands fit L2.  When
            # they do, a second read of B is a hit and allocating wins; when
            # they do not, they evict each other and bypass wins.  Only the side
            # the working set asks for is offered: the other one wins the
            # unfused ranking, where the bench keeps the operands hot, and then
            # loses the profiled run.
            operand_bytes = batch * elem_bytes * (m * k + k * n)
            if operand_bytes <= L2_BYPASS_FROM_BYTES:
                l2_pairs = ((0, 0),)
            else:
                l2_pairs = ((2, 2), (2, 4))
            for l2_mode_a, l2_mode_b in l2_pairs:
                add(1, block_m, block_n, block_k, l2_mode_a, l2_mode_b,
                    False, 0)
                # The pipelined twin, which _persistent_config drops unless the
                # plan streams A.
                add(1, block_m, block_n, block_k, l2_mode_a, l2_mode_b,
                    False, 0, pipelined=True)
            # Shaved plan under the pair that the shape actually wants, one
            # slot rather than one per L2 mode.
            add(1, block_m, block_n, block_k,
                l2_pairs[0][0], l2_pairs[0][1], False, 0, shave=1)

            # The same tile with N split across tasks instead of walked inside
            # one.  A is now the operand read more than once, so it is the one
            # offered a cache: mode 0 leaves it allocating in L2, where the
            # neighbouring cores holding the same batch's other N groups find
            # it.  B keeps no-allocate either way, being read once, so the pairs
            # that no-allocate A are not offered at all.
            groups = _n_group_candidate(
                (n + block_n - 1) // block_n,
                batch * ((m + block_m - 1) // block_m),
            )
            if groups > 1:
                # B is kept cached only while A+B still fits L2; past that only
                # A is, an N-split of a large working set having been measured
                # at half speed with both allocating.
                if operand_bytes <= L2_BYPASS_FROM_BYTES:
                    ng_pairs = ((0, 2), (0, 0))
                else:
                    ng_pairs = ((0, 2),)
                for l2_mode_a, l2_mode_b in ng_pairs:
                    add(1, block_m, block_n, block_k, l2_mode_a, l2_mode_b,
                        False, 0, n_groups=groups)
                    # An N-split task walks one N tile, so a streamed panel is
                    # read once either way and residency buys nothing.
                    add(1, block_m, block_n, block_k, l2_mode_a, l2_mode_b,
                        False, 0, n_groups=groups, pipelined=True)

    return configs


def _get_npu_bmm_batched_configs(
    batch: int,
    m: int,
    n: int,
    k: int,
    elem_bytes: int = DEFAULT_ELEM_BYTES,
) -> List[Dict[str, Any]]:
    """Configs for the batched template: the grouped regime.

    The regime is selected from the on-chip capacities rather than from a shape
    table, so a shape outside the tuned set still gets an answer.  All of the
    following must hold, and the list comes back empty otherwise, leaving the
    caller with the persistent or the generic template:

      * ``2 * (a_tile + b_tile) <= L1``  (double-buffered L1 staging fits)
      * ``c_tile <= L0C``                (accumulator fits)
      * ``block_q > 1``                  (at least two batches per group)
      * ``batch % block_q == 0``         (batch axis never ragged)
    """
    # One tile covers the whole matmul in this regime, so the tile is just the
    # shape rounded up to what tl.arange can express.
    block_m, block_n, block_k = _next_pow2(m), _next_pow2(n), _next_pow2(k)

    a_tile = _fractal_rows(block_m) * block_k * elem_bytes
    b_tile = block_k * block_n * elem_bytes
    c_tile = _fractal_rows(block_m) * block_n * ACCUM_BYTES

    # MarkMultiBuffer double-buffers the L1 staging buffers, so budget 2x.
    l1_per_batch = 2 * (a_tile + b_tile)

    if l1_per_batch > L1_BYTES or c_tile > L0C_BYTES:
        return []

    # Capacity is not the binding constraint -- the group loop is.  One group
    # is a single nd2nz -> nd2nz -> mmad -> fixpipe chain with nothing to
    # overlap against; MIN_WAVES groups per core reproduces the knee where
    # overlap stops paying for per-group fixed cost.
    max_q = min(L1_BYTES // l1_per_batch, L0C_BYTES // c_tile)
    max_q = min(max_q, max(1, batch // (MIN_WAVES * NUM_CUBE_CORES)))

    # BLOCK_Q must stay a power of two (tl.arange) and divide BATCH evenly,
    # otherwise every block pays for a batch mask and its linalg.fill.
    block_q = 1
    while block_q * 2 <= max_q and batch % (block_q * 2) == 0:
        block_q *= 2

    if block_q <= 1:
        return []

    programs, groups_per_program, ragged = _split_groups(batch // block_q)

    return [{
        # The group one mmad covers: BLOCK_Q batches of the whole BLOCK_M by
        # BLOCK_N tile, reduced over the whole of BLOCK_K at once.
        "BLOCK_Q": block_q,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        # The run of groups one program walks, and whether the last programs
        # come up short of it, which costs a trip count and no mask.
        "GROUPS_PER_PROGRAM": groups_per_program,
        "RAGGED_GROUPS": ragged,
        # Grid dimension: one program per run of groups.  Stored in meta so
        # npu_bmm_batched_grid can read it without re-deriving the split.
        "N_GROUP": programs,
        # Copies of the A/B staging buffers, so the next group's nd2nz runs
        # under this group's mmad instead of waiting on the same L1 slot.
        "L1_COPIES": L1_BUFFER_COPIES,
        # Whether an axis overhangs its rounded-up tile and needs masking.
        "MASKED_M": m % block_m != 0,
        "MASKED_N": n % block_n != 0,
        "MASKED_K": k % block_k != 0,
        # Off for the same reason as in the persistent template, and
        # additionally because BatchL1Mmad updates the flag only on the batch's
        # final mmad.
        "unit_flag": False,
        # As in the persistent template: the depth the loops run ahead, and a
        # warp count the cube path does not vary.
        "num_stages": 4,
        "num_warps": 4,
        "ALLOW_HF32": matmul.allow_hf32,
        "ACC_TYPE": "tl.float32",
        # This is the only template emitting a rank-3 tl.dot, and the option is
        # what keeps it one batched mmad macro with both operands resident in
        # L1.  Not a tuning knob: without it the rank-3 loads lower to a 2-D L1
        # load the template library has no entry for, so the kernel fails to
        # link.
        "enable_hivm_batch_matmul": True,
    }]


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

    The generic template is always offered, so the Triton path is never left
    without a choice, including on dynamic shapes.  The two specialised
    templates need a known batch count and are each offered only where the
    shape lands in their regime, which their config generators decide from the
    on-chip capacities: batched for small per-batch working sets over many
    batches, persistent for large single matmuls.
    """
    input_nodes = [mat1, mat2]

    candidates: List = [(npu_triton_bmm_template, _get_npu_bmm_configs(m, n, k))]

    if batch:
        # Both specialised regimes budget L1 and L0 in bytes, so they need the
        # operand size.  dtype_to_bytes answers 0 for anything it does not know,
        # and 0 would make every capacity check pass.
        elem_bytes = dtype_to_bytes(mat1.get_dtype()) or DEFAULT_ELEM_BYTES
        batched_configs = _get_npu_bmm_batched_configs(batch, m, n, k, elem_bytes)
        if batched_configs:
            candidates.append(
                (npu_triton_bmm_batched_template, batched_configs)
            )
        # The two regimes are not exclusive.  Treating them as such hid the
        # persistent template on any shape whose A+B tiles fit L1, where the
        # batched one wins the dispatch and then pads M or N up to a power of
        # two; offering both lets autotune take the narrower persistent tile
        # instead.
        want_persistent = (not batched_configs) or (m & (m - 1)) or (n & (n - 1))
        if want_persistent:
            persistent_configs = _get_npu_bmm_persistent_configs(
                batch, m, n, k, elem_bytes)
            if persistent_configs:
                candidates.append(
                    (npu_triton_bmm_persistent_template, persistent_configs)
                )

    for template, configs in candidates:
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
            except Exception as e:
                log.debug(
                    "Failed to generate %s choice with config %s: %s",
                    template.name,
                    cfg,
                    e,
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
        # The bmm template handles the batch dimension via tl.program_id(1)
        # and supports epilogue fusion via {{store_output}}.
        if is_nonzero and use_triton_template(layout):
            try:
                # The persistent template bakes the batch count into its loop
                # bounds, so it is only offered on static shapes; 0 tells
                # add_npu_triton_bmm_choices to skip it.
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
            except Exception as e:
                log.warning("Failed to add NPU triton bmm template choices: %s", e)

        if len(choices) == 0:
            log.warning("No choices for GEMM, using ATen backend as fallback")
            choices.append(aten_bmm.bind((mat1, mat2), layout))

        return autotune_select_algorithm("bmm", choices, [mat1, mat2], layout)