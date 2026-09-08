# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Central configuration for the triton_experimental backend.

Each tunable is a typed module-level default; ``install_config_module`` (torch's
own machinery) turns this module into a live config object with attribute access,
``config.patch(...)`` overrides, and backend-hash serialization::

    from torch_npu._inductor.triton_experimental import config
    if config.rsplit_outer: ...
    with config.patch(rsplit_outer=False): ...

No env-var layer: this object is the single source of truth. All gates are read
in the main process (codegen + in-process autotune); compile workers only build
already-generated source, so no gate crosses the subprocess boundary.
"""

import sys
from typing import Optional

from torch.utils._config_module import install_config_module


# =====================================================================
# Feature gates -- default ON. Guarded by ``if config.<name>:``.
# =====================================================================

# Cross-core r-axis split for OUTER reductions (40-core partial + combine).
rsplit_outer: bool = True

# Rewrite ``.to(tl.int1)`` load casts as ``!= 0`` (the packed-i1 trunc path can
# UB-OOB on Ascend).
rewrite_int1_cast_as_ne: bool = True

# Register lane-slice (extract_slice) rewrite for stride-k select loads, plus its
# strided-store companion.
select_extract_slice: bool = True
select_extract_slice_strided: bool = True

# A5 (910_95) ONLY: lower aten.embedding as a CANN register row gather
# (``extension.custom("__builtin_index_select", ...)``) via ops.index_select,
# instead of the upstream indirect ``tl.load(weight + H*idx + h)``. The extension
# op exists only on A5; gated by device_props.is_a5(), off elsewhere.
cann_index_select: bool = True

# Fold a fused ``(a + b) % c`` index so a divisor==1 x-tree axis stays linear.
fold_fused_mod: bool = True

# Fold a dual-view r-node index (mirrors fold_dual_decomp on the x-tree).
fold_dualview_rnode: bool = True

# Reindex a strided pointwise x-axis into a linear loop nest (pure reindex).
split_strided_pointwise: bool = True

# Keep a permuted store out of a fusion (operand-side permute instead).
no_fuse_permuted_store: bool = True

# Transposed dual-view x-node fold (axes a pure permutation of basis).
fold_transposed_xnode: bool = True

# Dual-decomposition fold and its flat-r-node companion.
fold_dual_decomp: bool = True
fold_flat_rnode: bool = True

# Expand div/mod in index formulas so axes merge at the upper layer.
expand_divmod: bool = True

# Collapse the x-tree where legal.
collapse_xtree: bool = True

# Emit tile-1 strided axes as scalar odometer offsets (no rank slot), and allow
# it inside reductions.
scalar_odometer_axes: bool = True
scalar_odo_in_reduction: bool = True

# Reduction-tree real-block autotile (dynamic-shape r0_ path).
rtree_real_block_autotune: bool = True

# Redirect a root-modulo axis split onto the stride-1 leaf.
split_root_modulo: bool = True

# Split a fused REDUCTION axis carrying ModularIndexing into inner/outer sub-nodes
# so ``r % s`` folds to the inner axis and the load address becomes affine;
# without it the modulo stays a scalar gather.
split_reduction_modulo: bool = True

# Skip re-adding a dense broadcast_to when the mask axes are already covered.
skip_redundant_broadcast: bool = True

# Realize a strided slice input into a contiguous buffer.
realize_strided_slice_input: bool = True

# Realize a sub-16B tail broadcast into its own contiguous buffer.
realize_tail_bcast: bool = True

# Reduce XBLOCK by input stride (input-stride-priority reduction tiling).
reduce_xblock_by_input_stride: bool = True

# Unify real_block==tile for eligible free axes (flat greedy tiling).
unify_block: bool = True

# Pad the innermost reduction block up to a multiple of 8.
pad_min_block_to_8: bool = True

# Odometer-offset optimization for free-axis iteration.
odometer_opt: bool = True

# Reassociate ``sum(X + broadcast(Y))`` -> ``sum(X) + broadcast(sum(Y))``
# (linear-op reassociation, not bit-identical).
reassociate_sum_of_add: bool = True

# Elide an int->float->int round-trip.
elide_int_float_int: bool = True

# Fold ``Max(1, size)`` in loop-merge index formulas so conv-output axes merge.
fold_max1_in_loop_merge: bool = True

# Downcast fp64/i64 boundary once, deduped across fused kernels.
int64_boundary_cast: bool = True
dedup_downcast: bool = True

# Group-dispatch across the 40 cores.
group_dispatch: bool = True

# --- default-ON gates whose NAME reads as "disable X"; guarded by
#     ``if not config.<name>:`` (True == feature active). ---

# Disable recursive dict tag guards (torch_npu monkeypatch conflict workaround).
disable_recursive_dict_tag_guards: bool = True
# Disable the shape-padding pad_mm family.
disable_pad_mm: bool = True
# Disable the add+mm -> addmm fusion.
disable_addmm_fusion: bool = True

# Auto-preload libmspti.so, and use the mspti device-time autotune bench path.
enable_mspti: bool = True
autotune_mspti: bool = True

# --- shared triton/autotune gates (read by our Python; not passed to triton) ---

# Linearize triton codegen (the NPU flat-index codegen path).
codegen_linearize: bool = True
# Fold a >65535 grid down so all blocks run in parallel.
all_blocks_parallel: bool = True
# Enhanced autotune config sweep.
autotune_enhance: bool = True
# ON: default 1D pointwise and reduction path uses the legacy block-list cascade instead of the
# formula-driven _pw1d_formula_configs (A/B fallback; off by default).
autotune_fallback: bool = False


# =====================================================================
# Feature gates -- default OFF (experimental / opt-in).
# =====================================================================

# Drop the dead accumulator ``tl.where`` guard in reductions (SSA proof over
# per-load pad-lane facts, not text matching).
elide_reduction_where: bool = True

# Promoted rtree shape fix: after the flat-loop rewrite, re-align masked-load ptrs
# (bar-shaped index vs block-shaped mask) and store offsets (pre-promotion rank vs
# rank-N value) with the promoted block shape. triton-ascend rejects the implicit
# ptr-vs-mask broadcasts upstream Triton tolerates.
promoted_rtree_shape_fix: bool = True

# Inject ``care_padding=False`` on masked loads.
inject_care_padding: bool = False

# Refactor expanded conv-output store strides onto precomputed ks.
refactor_clamp_stride: bool = False

# =====================================================================
# Permute-gather strided-reduction rewrite (opt-in: enable_permute_gather).
# All tuning thresholds live in this section; codegen keeps only the UB-budget
# formula constants (_PG_UB_*), matching upstream practice of keeping formula
# internals module-local (e.g. TRITON_MAX_BLOCK in torch/_inductor codegen).
# =====================================================================

# Realize a permute+gather into a contiguous buffer at lowering. Default ON: a
# non-unit inner stride pushed onto the reduction axis (e.g. T5 fwd softmax with a
# relative-position bias, logical [heads,q,k] over [q,k,heads] storage) degrades to
# a scalar gather on Ascend -- ~55ms/iter. Realizing the permute removes the gather
# (T5 fwd softmax pos_bias ~50ms -> ~0.46ms, 110x). See lowering.npu_permute for the
# guards that keep transpose-for-matmul (harmless non-unit inner stride) on the fast
# no-realize path.
realize_permute_gather: bool = True

# Codegen a contiguous-DMA + tl.gather for a permute that pushes a non-unit stride
# onto the reduction axis, instead of the strided tl.load (scalar gather on Ascend).
# Keeps the permute as a zero-copy logical view (no realize buffer) and rewrites the
# consumer load into a contiguous burst DMA of the whole row, then a register-level
# gather into the logical tile. Requires a reduction axis with unit-input-coeff interior
# (single-interior geometry, e.g. bias[Sq,Sk,H] permute(2,0,1) reduce over Sk); other
# shapes fall back to the strided load. OFF by default; opt-in per compile.
enable_permute_gather: bool = False

# Why 256: the measured fp32 gather/trans crossover sits between stride_r=24
# (96B: gather wins) and 64 (256B: gather loses, its flat tile overflows UB at
# R0=128); 64 fp32 = 256B is also half the 910B2 segment-prefetch granularity
# knee (512B). NOT a DMA alignment boundary -- the flat DMA is alignment-agnostic.
#
# Benefit gate for the permute-gather rewrite, in bytes of transpose granularity
# (permuted inner stride * elemsize). Below this (stride_bytes < gate) the
# register gather wins: its flat contiguous DMA is alignment-agnostic and the
# flat tile stays under UB (H<=63 fp32). At/above the gate the layout goes to
# the trans mode (block_ptr + tl.trans, tails via boundary_check): a gather flat
# tile would overflow UB (measured H=64: trans 65.7us vs gather R0=64 93.2us,
# R0=128 compile fail) and trans beats the realized transpose (two-kernel 219us).
# Measured crossover (fp32, per-iteration device time): stride_r=24/96B gather
# 46.7us < strided 66us < eager 58us; stride_r=64/256B gather 99us ~ strided
# 102us > eager 67us; stride_r=128/512B gather 324us > strided 190us. At the
# same threshold lowering keeps the permute zero-copy (view for the rewrite) vs
# realizing it (fast strided fallback).
permute_gather_stride_gate_bytes: int = 256

# Why 64: 64 fp32 = 256B chunk = the gate's DMA efficiency unit; pow2 and a
# 32B-multiple (hard constraint below), keeping the XBLOCK pin autotune-legal
# while boundary_check absorbs tails.
#
# Static-trans int-axis chunk width, in ELEMENTS (fp32: 64 = 256B chunk, i.e.
# the same boundary as the gate above). Static trans pins XBLOCK =
# min(stride_r, ktile) instead of stride_r, so the interior axis is chunked
# across programs (x1_blocks = ceil(H/ktile)) through the greedy tile chain +
# group dispatch -- the same mechanism dynamic-H trans already uses, with tails
# kept exact by boundary_check. Decouples the R0_BLOCK UB cap from H (R0 stops
# collapsing as H grows: H=4096 R0 1 -> 64, 300 -> 5 r-trips). Must be a
# multiple of 32B/elemsize (fp32: 8), else tile_align rounding breaks the
# forced-tiling eval and the rewrite falls back to strided.
permute_gather_ktile: int = 64

# Dynamic-H (symbolic reduction stride) fallback: the gather index must be
# compile-time affine in stride_r (rejected by the geometry's static-int gate), so
# dynamic H can only ride the trans mode's block_ptr + boundary_check, which is
# shape-generic. OFF -> the strided-load fallback for dynamic H.
permute_gather_dynamic_trans: bool = True

# Why 4096: TRITON's max_block -- the same cap family as upstream
# TRITON_MAX_BLOCK (see npu_triton_config_reduction).
#
# Gather-mode XBLOCK cap: the gather rewrite pins XBLOCK = stride_r (the
# interior axis runs the full head in one tile), and the reduction heuristic
# caps XBLOCK at TRITON's max_block (4096, see npu_triton_config_reduction). A
# stride_r above that cannot be gathered -> permute_gather_mode returns None
# for the gather branch, so lowering realizes the permuted layout (the
# pre-dispatch behavior) instead of compiling an "XBLOCK too large" kernel for
# huge H (measured H=16384 fp32). The trans branch is NOT capped: it pins
# XBLOCK = min(stride_r, permute_gather_ktile), which stays 64 for any H, so
# huge H goes through trans with the int axis chunked across programs.
permute_gather_max_xblock: int = 4096


def permute_gather_mode(stride_r, elemsize):
    """Dispatch the permute rewrite for a static permuted inner stride of
    ``stride_r`` elements of ``elemsize`` bytes: "gather" / "trans" / None
    (fall back to realize). Per-branch caps and mode semantics: see the flags
    above. Dynamic (symbolic) strides are dispatched in the codegen geometry,
    which can see the symbolic reduction coefficient. Reads the live config
    object (install_config_module moves the typed defaults off module globals
    into instance attributes)."""
    from torch_npu._inductor.triton_experimental import config as _cfg
    stride_bytes = stride_r * elemsize
    if stride_bytes < _cfg.permute_gather_stride_gate_bytes:
        if stride_r > _cfg.permute_gather_max_xblock:
            return None  # gather pins XBLOCK = stride_r; capped at max_block
        return "gather"
    return "trans"

# Route the MASK-COMPOSITE softmax (aten._safe_softmax, produced from
# transformers-style causal-mask + softmax patterns) through aclnn instead of
# Triton fusion: the fused Triton kernel materializes [B,H,S,S] masks and
# measured ~2x slower than the eager aclnn sequence (TrOCR: 10.3 ms/iter of
# mask-materialization device time). Eager dispatch of _safe_softmax runs the
# composite down to plain softmax → aclnnSoftmax (bit-identical, verified).
# Plain softmax KEEPS its Triton fusion (profitable for e.g. BertForMaskedLM).
safe_softmax_aclnn_fallback: bool = True

# Route the log_softmax family through aclnnLogSoftmax: the loss-path Triton
# log_softmax kernel is slower than aclnnLogSoftmax and its fusion drags a
# gather decomposition along (TrOCR: Gather_AsStrided +2.9 ms/iter).
log_softmax_aclnn_fallback: bool = True

# Plain-softmax size routing: rows (reduction width) up to this bound keep the
# Triton fusion; wider rows fall back to aclnnSoftmax (see decomposition.py
# _override_plain_softmax_width_decomp: >1024 loses persistent-reduction
# eligibility and TE keeps split_reductions off, wide rows degrade to a serial
# per-row scan, 2.5-2.8x slower at vocab widths). 0 disables routing (always
# Triton).
softmax_aclnn_max_fuse_numel: int = 256
# ---- Clone-fold contract -------------------------------------------------
# rescue_rules.fold_verdict (Line 1, lowering.npu_clone) decides fold vs
# materialize from tile-independent physics; npu_header._npu_pointwise_
# tile_contract (Line 2) enforces the stride-1-survives-tiling premise and
# reports violations.  Every default follows the cost asymmetry: an
# unrescued fold costs 30-130x (mobilevit_s 27-48 ms/launch vs ~1 ms
# eager), over-materializing 1.5-3x -- so doubt always materializes.
#   "contract" (default) | "fold-all"/None = upstream lazy | "materialize-all"
clone_policy: Optional[str] = "contract"

# Symbolic layouts are judged by the same verdict over the guard-free
# substrate (statically_known_* / optimization_hint; int()/guard_int -- which
# broke mark_dynamic historically -- is never used).  False = old skip-to-fold.
clone_contract_dynamic: bool = True

# Family-R cost gates (L2 transpose trips / L4 min misplaced-axis extent).
# END-TO-END-MEASURED constants (probe-v2 decision table) -- do not tighten
# analytically; L4 keeps the strict form (the stride-1-axis exemption
# regressed mobilevit_s to 0.28x).
clone_rescue_bet_max_swaps: int = 2
clone_rescue_bet_min_axis: int = 8

# Line 2 switch (False = baseline greedy order, bisect arm) and the
# example-shape second ballot for unprovable numeric gates / undecidable
# orders (verdicts may track the shape bucket on that tier; both outcomes
# are always numerically correct).
pointwise_tile_contract: bool = True
clone_dynamic_hint_tier: bool = True

# "log" (default; None = log) = WARN on fold bets that landed off ①/②;
# "strict" = additionally raise (CI/adversarial); "off" = silent.
clone_contract_check: Optional[str] = "log"


# Reduction-tree real-block promotion (nested scalar r-loops -> real-block tile).
rtree_real_block: bool = True

# Flatten small outer r-nodes.
flatten_small_outer_rnodes: bool = False

# Compare an int arange/numel mask in fp32 (only when numels stay below ~8M).
mask_cmp_fp32: bool = False


# =====================================================================
# Integer tuning knobs.
# =====================================================================

# Scalar-odometer emission budget (max product of tile-1 axis extents).
scalar_odo_budget: int = 1024

# Max reads a single fused kernel may carry.
max_fused_reads: int = 24

# Reduction-tree real-block tile cap.
rtree_real_block_cap: int = 2048

# Below this inner-run vector width, realize a tail broadcast.
tail_bcast_min_vec: int = 8

# Upper bound on a pointwise 2D tile (XBLOCK*YBLOCK); autotune prunes over-UB.
pointwise_tile_max: int = 65536

# Innermost stride-1 axis alignment; <=0 disables alignment.
tile_align: int = 8

# Pinned balanced-tile target (0 = auto / runtime XBLOCK).
balanced_target: int = 0

# --- autotune-bench knobs (read live at bench time) ---
mspti_warmup: int = 5
mspti_active: int = 20
event_bench_max_inner: int = 256
event_bench_max_groups: int = 25


# =====================================================================
# Float tuning knobs.
# =====================================================================

# Target per-group wall time (ms) for the NPU-event autotune bench.
event_bench_target_ms: float = 8.0


# =====================================================================
# Sentinel / optional knobs (None == unset, caller decides).
# =====================================================================

# TEMP diagnostic: pin a single (XBLOCK, R0_BLOCK) as "x,r"; None = full sweep.
pin_xr: Optional[str] = None

# Override Ascend A5 (910_95) detection; None = auto-detect by soc version.
force_is_a5: Optional[bool] = None


# =====================================================================
# Debug toggles.
# =====================================================================

# Master inductor debug (config dumps, autotune config lists, etc.).
debug: bool = False
# Under debug, route compile-worker stdout here so NPU debug prints reach the
# terminal (see _route_worker_logs_for_debug).
worker_log_path: str = "/dev/stdout"
# Per-launcher triton device-time debug.
triton_debug: bool = False


# =====================================================================
# Codegen toggles (sub-config namespace, e.g. config.npu_triton.<x>).
# =====================================================================
class npu_triton:
    # Whether to upcast float16 / bfloat16 to float32 in triton codegen (Experimental).
    codegen_upcast_to_fp32 = True


# =====================================================================
# VF Fusion Controls (owned by the bisheng compile path).
# =====================================================================

# VF Fusion 总开关: 合并多个独立 AIV (Vector) 函数为一个
# True = 启用, False = 关闭 (默认)
enable_vf_fusion: bool = False

# VF Fusion 策略: 控制融合激进度
# None = 不设置 (UBTuner 自动选择)
# "ub-aware-op" = UB 感知模式 (保守, 低 UB 消耗)
# "max-parallel" = 最大并行模式 (激进, 高 UB 消耗, 更高性能)
vf_fusion_mode: Optional[str] = None

# VF Merge 级别: 函数内部 vector 操作合并程度
# None = 不设置 (后端默认值 1)
# 0 = 不合并, 1 = 标准合并
vf_merge_level: Optional[int] = None


# adds .patch(), .save_config(), attribute access, backend-hash serialization
install_config_module(sys.modules[__name__])
