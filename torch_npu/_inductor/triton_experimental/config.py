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

# Route a step-1 slice_scatter through the cat_loop sub-region store instead of
# the upstream where-blend. slice_scatter(x, src, dim, A, B) == cat([x[:A], src,
# x[B:]]); upstream blends src at index (idx-A), whose negative base offset the
# NPU backend miscompiles once the scattered axis splits across blocks (wrong
# numerics / "not runnable"). cat_loop stores each input at output coord lo_i+c
# from a 0-based local load under mask c<size_i, avoiding the negative base. Only
# the buggy shape (step==1, start>0, backed bounds) is rerouted.
slice_scatter_via_cat_loop: bool = True

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


# =====================================================================
# Feature gates -- default OFF (experimental / opt-in).
# =====================================================================

# Drop the dead accumulator ``tl.where`` guard in reductions (SSA proof over
# per-load pad-lane facts, not text matching).
elide_reduction_where: bool = True

# Inject ``care_padding=False`` on masked loads.
inject_care_padding: bool = False

# Refactor expanded conv-output store strides onto precomputed ks.
refactor_clamp_stride: bool = False

# Realize a permute+gather into a contiguous buffer at lowering. Default ON: a
# non-unit inner stride pushed onto the reduction axis (e.g. T5 fwd softmax with a
# relative-position bias, logical [heads,q,k] over [q,k,heads] storage) degrades to
# a scalar gather on Ascend -- ~55ms/iter. Realizing the permute removes the gather
# (T5 fwd softmax pos_bias ~50ms -> ~0.46ms, 110x). See lowering.npu_permute for the
# guards that keep transpose-for-matmul (harmless non-unit inner stride) on the fast
# no-realize path.
realize_permute_gather: bool = True

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
