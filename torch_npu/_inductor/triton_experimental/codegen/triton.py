# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Huawei Technologies Co., Ltd
# Copyright (c) 2013 the respective contributors
#
# Licensed under the Apache-2.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch._inductor.codegen.triton import (
    TritonScheduling,
    TritonKernel,
    TritonKernelOverrides,
    BlockPtrOptions,
    TritonCSEVariable,
    IndexingOptions,
    texpr,
    log,
    is_welford_reduction,
    TritonSymbols,
)
from torch.utils._sympy.symbol import symbol_is_type, prefix_str, SymT
from torch._inductor.codegen.multi_kernel import MultiKernel
from torch._dynamo.utils import counters
from torch._inductor.codegen.simd import (
    IterationRangesRoot,
    IterationRangesEntry,
)
from torch._inductor.codegen.simd_kernel_features import (
    SIMDKernelFeatures,
    DisableReduction,
    EnableReduction,
)
from torch._inductor.utils import (
    IndentedBuffer,
    Placeholder,
    prefix_is_reduction,
    sympy_subs,
)
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch._inductor.codegen.common import (
    SizeArg,
    CSEVariable,
    ArgName,
    ConstexprArg,
)
from torch._inductor.codegen.wrapper import SymbolicCallArg
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from typing import (
    cast,
    Dict,
    Any,
)

from .. import config as ncfg
import re
import math
import sympy
import contextlib

from torch._inductor import config
from .. import device_props
import torch


triton_codegen_linearize = ncfg.codegen_linearize
# When set, drop the dead accumulator ``tl.where`` guard in a static single-tile
# reduction. Coupled 1:1 with npu_triton_heuristics.npu_elide_reduction_where,
# which pins R0_BLOCK==rnumel so the reduction-axis mask is provably always-true.
npu_elide_reduction_where = ncfg.elide_reduction_where


def _is_flat_rloop_header(stripped: str, prefix: str) -> bool:
    """True if `stripped` is an upstream flat reduction-loop header for `prefix`
    (``for <prefix>offset in [tl.]range(0, ...``; accept both range forms). The
    ``(0,`` start avoids catching r-split-rewritten ``range(<lo>, <hi>, ...``."""
    return (
        stripped.startswith(f"for {prefix}offset in range(0,")
        or stripped.startswith(f"for {prefix}offset in tl.range(0,")
    )

def _npu_scalar_odometer_axis_names(kernel, tree):
    """Free x-tree axis names that should be SCALAR odometer axes, not register
    tile dims.

    The greedy tiler gives each free axis tile = min(size, rem); once the
    contiguous head fills the budget, every axis behind it gets tile==1. Those
    still emit as ``tl.arange(0,1)[...]`` register dims, inflating the tile rank
    (T5 transpose add -> rank-4 [1,1,y0,x3]) which BiShengIR can't collapse -> UB
    blowup + ~40x (4969us vs 116us rank-2). A tile-1 axis carries no width, so it
    belongs on the odometer as a scalar offset (real_block==1); since these are
    the strided axes it costs no bandwidth. Gated by scalar_odometer_axes: with
    >=2 free axes, walk most-contiguous-first under scalar_odo_budget; any axis
    whose tile hint is 1 (and all after it) becomes scalar; the head is never
    collapsed.
    """
    if not ncfg.scalar_odometer_axes:
        return set()
    if tree.is_reduction or getattr(kernel, "no_x_dim", False):
        return set()
    # Inside a reduction the free X-tree axes carry an input-stride priority chain
    # (xblock_priority_denom); the same tile-1 rank pathology applies, so collapse
    # here too. Safe: the divisor-ordered walk only collapses a trailing suffix, and
    # the priority denom references only higher-priority axes.
    _in_reduction = getattr(kernel, "inside_reduction", False)
    if _in_reduction and not ncfg.scalar_odo_in_reduction:
        return set()
    free_nodes = [n for n in tree.nodes.values()
                  if n.name not in getattr(tree, "tree_node_mapping", {})]
    if len(free_nodes) < 2:
        return set()

    def _divisor_hint(n):
        if isinstance(n.divisor, (int, sympy.Integer)):
            return int(n.divisor)
        try:
            return int(V.graph.sizevars.optimization_hint(n.divisor))
        except Exception:
            return 1 << 30

    def _size_hint(n):
        if isinstance(n.length, (int, sympy.Integer)):
            return int(n.length)
        try:
            h = int(V.graph.sizevars.optimization_hint(n.length))
            return h if h > 0 else None
        except Exception:
            return None

    try:
        budget = ncfg.scalar_odo_budget
    except ValueError:
        budget = 1024
    if budget < 1:
        budget = 1

    ordered = sorted(free_nodes, key=_divisor_hint)  # most-contiguous first
    scalar = set()
    rem = budget
    collapsed = False
    for idx, node in enumerate(ordered):
        s = _size_hint(node)
        if collapsed:
            # Budget exhausted: this and later axes tile 1 under the fixed budget.
            # But a real size hint > 1 means autotune can find an XBLOCK giving it a
            # non-unit tile, so don't pre-classify it as scalar; let the greedy path
            # handle tile = min(size_hint, remainder).
            if isinstance(s, int) and s > 1:
                continue
            scalar.add(node.name)
            continue
        if s is None:
            # Unknown size eats the remaining budget; nothing collapses on it.
            rem = 1
            collapsed = rem <= 1 and idx + 1 < len(ordered)
            continue
        tile = s if s < rem else rem
        # Head axis is never collapsed even if its own size is 1 (degenerate).
        if idx > 0 and tile <= 1:
            # Same guard: a non-trivial size hint means autotune can find an
            # XBLOCK where this axis gets a real tile. Skip scalar classification.
            if isinstance(s, int) and s > 1:
                rem = rem // tile if tile > 0 else 1
                if rem <= 1:
                    collapsed = True
                continue
            scalar.add(node.name)
            continue
        rem = rem // tile if tile > 0 else 1
        if rem <= 1:
            collapsed = True
    return scalar


def _npu_rewrite_promoted_rtree_body(kernel, real_sizes, real_ndim):
    """Rewrite the assembled reduction body for promoted r-trees. Per tree: turn
    the flat ``for r0_offset ...`` loop into full-residency aranges (each free
    r-node N gets ``N = tl.arange(0, real_block_N)[slot]`` + ``Nmask``, combined
    ``r0_mask``), hoist the real_block_N constexpr defs before the group loop, and
    reshape-collapse each ``tl.sum(_acc, dim)`` so the r-slots merge to one axis.
    Handles only the fully-resident case (guaranteed by the promotability gate)."""
    sv = V.graph.sizevars
    for tree in kernel.range_trees:
        if not tree.is_reduction:
            continue
        ordered_names = kernel._npu_rtree_promoted.get(tree.prefix)
        if not ordered_names:
            continue
        prefix = tree.prefix  # e.g. "r0_"
        name_to_node = {n.name: n for n in tree.nodes.values()}
        vtd = getattr(tree, "var_tensor_dims", {}) or {}

        # Per-node real_block constexpr defs + arange/mask body lines.
        # Slot broadcast: ``[None, ..., :, ..., None]`` with ``:`` at the
        # node's tensor_dim slot, one entry per real_ndim.
        def _slot_bcast(slot):
            parts = ["None"] * real_ndim
            if 0 <= slot < real_ndim:
                parts[slot] = ":"
            return "[" + ", ".join(parts) + "]"

        r_slots = sorted(vtd[nm] for nm in ordered_names if nm in vtd)
        dynamic = kernel._npu_rtree_dynamic.get(prefix, False)
        tiles = kernel._npu_rtree_tile
        node_dyn = getattr(kernel, "_npu_rtree_node_dynamic", {})
        # Derive dynamic r-node tiles from the reduction block (R0_BLOCK) so
        # autotune explores the reduction tile (mirrors X-axis real_block = XBLOCK
        # // d). Guarded to "r0_" and rtree_real_block_autotune; else use the hint
        # literal in _npu_rtree_tile.
        r_block_name = f"{prefix.upper()}BLOCK"  # "r0_" -> "R0_BLOCK"
        autotune_tiles = (
            dynamic and prefix == "r0_"
            and ncfg.rtree_real_block_autotune
        )
        # Resident (static-length) nodes consume their full length from the
        # R0_BLOCK budget; dynamic nodes share R0_BLOCK // static_prod.
        static_prod = 1
        for _nm in ordered_names:
            if not (dynamic and node_dyn.get(_nm, True)):
                static_prod *= int(sv.optimization_hint(name_to_node[_nm].length))
        # Per dynamic node, build an R0_BLOCK-derived constexpr tile, allocated
        # INNER-FIRST (divisor ascending) so an outer node's denominator can
        # reference an inner {nm}_blk; clamped to the length hint and floored at 1.
        dyn_blk_expr = {}     # nm -> RHS expression (no "nm_blk : tl.constexpr =")
        if autotune_tiles:
            def _rdiv(n):
                d = name_to_node[n].divisor
                return (int(d) if isinstance(d, (int, sympy.Integer))
                        else int(sv.optimization_hint(d)))
            dyn_names = [nm for nm in ordered_names if node_dyn.get(nm, True)]
            inner_first = sorted(dyn_names, key=_rdiv)  # ascending divisor
            denom_terms = [str(static_prod)] if static_prod > 1 else []
            for nm in inner_first:
                lh = int(sv.optimization_hint(name_to_node[nm].length))
                denom = "*".join(denom_terms) if denom_terms else "1"
                budget = (f"({r_block_name} // ({denom}))" if denom != "1"
                          else r_block_name)
                e = f"({budget}) if ({budget}) <= {lh} else {lh}"
                e = f"({e}) if ({e}) > 0 else 1"
                dyn_blk_expr[nm] = e
                denom_terms.append(f"{nm}_blk")
        static_defs = []     # real_block_{nm} defs (literal lengths)
        dyn_defs = []        # {nm}_blk defs, emitted inner-first (forward refs)
        arange_lines = []    # static nodes: N = tl.arange(...)[slot]; Nmask = ...
        loop_nodes = []      # dynamic nodes: (nm, slot, tile, "{nm}numel")
        mask_terms = []
        blk_node_names = []  # LHS names of blk_defs (for the hoist anchor search)
        # Per-node static/dynamic split: a static-length node is fully resident
        # (arange, no loop) even in a dynamic tree; only symbolic-length nodes get a
        # tile-loop. ordered_names is divisor-descending (loop_nodes outer-first).
        for nm in ordered_names:
            node = name_to_node[nm]
            slot = vtd[nm]
            is_node_dyn = dynamic and node_dyn.get(nm, True)
            if is_node_dyn:
                # Constexpr tile width + accumulation loop covering the full length,
                # masked per tile. Bound is the runtime {nm}numel arg for a symbolic
                # node, or the literal length for a static over-cap node we tile.
                tile = int(tiles[nm])
                blk_node_names.append(f"{nm}_blk")
                bound = str(int(node.length)) if isinstance(node.length, (int, sympy.Integer)) else f"{nm}numel"
                loop_nodes.append((nm, slot, tile, bound))
                mask_terms.append(f"{nm}mask")
            else:
                length = int(sv.optimization_hint(node.length))
                static_defs.append(f"real_block_{nm} : tl.constexpr = {length}")
                blk_node_names.append(f"real_block_{nm}")
                bcast = _slot_bcast(slot)
                arange_lines.append(
                    f"{nm} = tl.arange(0, real_block_{nm}){bcast}"
                )
                arange_lines.append(f"{nm}mask = {nm} < {length}")
                mask_terms.append(f"{nm}mask")
        # Emit dynamic _blk defs INNER-FIRST so chained denominators resolve;
        # fall back to the hint literal when autotune-derivation is disabled.
        if autotune_tiles:
            dyn_defs.extend(f"{nm}_blk : tl.constexpr = {dyn_blk_expr[nm]}" for nm in inner_first)
        else:
            dyn_defs.extend(f"{nm}_blk : tl.constexpr = {int(tiles[nm])}" for nm in ordered_names if dynamic and node_dyn.get(nm, True))  # noqa: B950
        # Static real_block defs reference only literals; dynamic defs reference
        # static_prod (literal) + inner dyn names — so dyn inner-first then any.
        blk_defs = dyn_defs + static_defs
        combined_mask = " & ".join(mask_terms)

        # Collapsed shape for the reshape before tl.sum: fold the contiguous r-slot
        # run into one product term IN PLACE (row-major reshape reinterprets memory,
        # so the fold must sit at the real slot order). r-slots are contiguous (gate)
        # but may sit anywhere — front or back. Walk slots in order, emit the
        # r-product at the first r-slot and each kept token at its own position.
        first_r = min(r_slots)
        last_r = max(r_slots)
        r_tokens = [real_sizes[s] for s in r_slots]
        r_product_tok = "*".join(r_tokens) if r_tokens else "1"
        collapsed_tokens = []
        for s in range(real_ndim):
            if s < first_r or s > last_r:
                collapsed_tokens.append(real_sizes[s])
            elif s == first_r:
                collapsed_tokens.append(r_product_tok)
            # first_r < s <= last_r: r-slot already folded into the product.
        # Reduction axis after collapse = the position of the folded r-run,
        # which is the count of kept slots preceding it. Since r-slots are
        # contiguous starting at first_r, every slot < first_r is a kept slot.
        collapsed_dim = first_r
        collapsed_shape = "[" + ", ".join(collapsed_tokens) + "]"
        # Post-sum resize: re-insert a singleton at EACH r-slot so the
        # result rank matches the kept-axis stores (x0 index is rank real_ndim).
        resize_parts = [":"] * real_ndim
        for s in r_slots:
            resize_parts[s] = "None"
        # The summed tensor has rank (len(keep_tokens)) — re-expand only the
        # r-slots as None; kept slots stay ':'.
        post_resize = "[" + ", ".join(resize_parts) + "]"

        # Split-reconstruction aliases (r0_1 = r0_2 + ks0*r0_3 over promoted
        # sub-nodes): the upstream body still emits r0_1 = r0_index, but promotion
        # deletes r0_index, so rewrite each to its reconstruction text (from
        # _npu_flat_rnode_subs) inside the loop where r0_2/r0_3 are in scope.
        recon_nodes = getattr(tree, "_npu_split_recon_nodes", None) or set()
        flat_subs = getattr(kernel, "_npu_flat_rnode_subs", None) or {}
        flat_recon = {nm: flat_subs[nm] for nm in recon_nodes if nm in flat_subs}

        _npu_apply_promoted_rtree_lines(
            kernel, prefix, blk_defs, blk_node_names, arange_lines,
            combined_mask, collapsed_shape, collapsed_dim, post_resize,
            r_slots, real_ndim,
            dynamic=dynamic, loop_nodes=loop_nodes, slot_bcast=_slot_bcast,
            flat_recon=flat_recon,
        )

def _npu_apply_promoted_rtree_lines(
    kernel, prefix, blk_defs, blk_node_names, arange_lines,
    combined_mask, collapsed_shape, collapsed_dim, post_resize,
    r_slots, real_ndim,
    dynamic=False, loop_nodes=None, slot_bcast=None, flat_recon=None,
):
    """In-place edit of kernel.body._lines for one promoted r-tree. Static mode:
    replace the flat ``for r0_offset`` loop with loop-free per-node aranges.
    Dynamic mode: replace it with nested tile loops (one accumulation loop per free
    r-node, each indexing a constexpr {nm}_blk tile); the accumulator persists and
    the reshape-collapse + sum runs once after the loops close."""
    loop_nodes = loop_nodes or []
    flat_recon = flat_recon or {}
    n_levels = len(loop_nodes)
    # Extra indent for the now-deeper innermost-loop body vs the flat-loop body.
    body_extra_indent = "    " * (n_levels - 1) if dynamic and n_levels else ""
    old_lines = kernel.body._lines
    new_lines = []
    inserted_defs = False   # hoist block constexpr defs before the group loop
    inside_rloop = False
    rloop_indent = ""

    # Patterns for stale lines to drop inside/around the r-loop.
    def _is_node_decomp(stripped):
        # ``r0_1 = (r0_index % 16)`` / ``r0_2 = r0_index // 16`` etc.
        return (
            (f"{prefix}index" in stripped)
            and (" % " in stripped or " // " in stripped)
            and stripped.split(" = ", 1)[0].strip().startswith(prefix)
        )

    for raw in old_lines:
        line = raw if isinstance(raw, str) else None
        if line is None:
            # Non-str deferred line (e.g. DelayReplaceLine). Inside the flat r-loop
            # we're eliminating, its indent follows the body (static: dedent 4;
            # dynamic: indent 4*(n_levels-1)) and its r0_mask -> combined mask.
            if inside_rloop:
                rep = raw
                if dynamic:
                    if body_extra_indent and hasattr(rep, "with_prefix"):
                        rep = rep.with_prefix(body_extra_indent)
                else:
                    # Dedent by 4 via slicing (DeferredLineBase.__getitem__).
                    cur = rep.line
                    if len(cur) - len(cur.lstrip()) >= 4:
                        rep = rep[4:]
                # Rewrite the r-mask term inside the deferred line text.
                if f"{prefix}mask & xmask" in rep.line:
                    rep = rep._new_line(rep.line.replace(
                        f"{prefix}mask & xmask", f"({combined_mask}) & xmask"
                    ))
                elif f"{prefix}mask" in rep.line:
                    rep = rep._new_line(rep.line.replace(
                        f"{prefix}mask", f"({combined_mask})"
                    ))
                new_lines.append(rep)
            else:
                new_lines.append(raw)
            continue
        stripped = line.strip()
        indent = line[:len(line) - len(stripped)] if stripped else ""

        # Hoist block constexpr defs before their first use: the group loop is in
        # the header (not kernel.body), so anchor on the first body line
        # referencing a block token (the accumulator tl.full) and insert above it.
        if not inserted_defs and any(
            bn in stripped for bn in blk_node_names
        ):
            new_lines.extend(f"{indent}{d}" for d in blk_defs)
            inserted_defs = True

        # Drop the stale ``rbase = tl.arange(0, R0_BLOCK)...`` and
        # ``r0_base = ...`` / ``rbase = r0_base`` / ``rnumel`` / ``RBLOCK``
        # scaffolding — the tile aranges / loops replace it.
        if (
            stripped.startswith("rbase = tl.arange")
            or stripped.startswith(f"{prefix}base = tl.arange")
            or stripped == f"rbase = {prefix}base"
            or stripped.startswith("rnumel = ")
            or stripped.startswith("RBLOCK: tl.constexpr")
            or stripped.startswith("RBLOCK : tl.constexpr")
        ):
            continue

        # Detect / remove the flat r-loop header; emit aranges (static) or the
        # nested tile-loop headers + per-tile aranges (dynamic) in its place.
        if not inside_rloop and _is_flat_rloop_header(stripped, prefix):
            inside_rloop = True
            rloop_indent = indent
            if dynamic:
                # Nested loop headers, outer-first (loop_nodes is divisor-descending).
                # Only symbolic-length nodes get loops; static nodes emit as
                # fully-resident aranges at the innermost level.
                inner_indent = indent
                for (nm, slot, tile, numel) in loop_nodes:
                    new_lines.append(
                        f"{inner_indent}for {nm}inner in range(0, {numel}, {nm}_blk):"
                    )
                    inner_indent += "    "
                # Per-tile dynamic index + mask at the innermost level.
                for (nm, slot, tile, numel) in loop_nodes:
                    bcast = slot_bcast(slot) if slot_bcast else ""
                    new_lines.append(
                        f"{inner_indent}{nm} = {nm}inner + tl.arange(0, {nm}_blk){bcast}"
                    )
                    new_lines.append(f"{inner_indent}{nm}mask = {nm} < {numel}")
                # Static (fully-resident) node aranges — same innermost level so
                # the combined mask + reshape below see every node in scope.
                new_lines.extend(f"{inner_indent}{al}" for al in arange_lines)
            else:
                new_lines.extend(f"{indent}{al}" for al in arange_lines)
            continue

        if inside_rloop:
            # End of loop body: a line indented <= the loop header.
            cur_indent = line[:len(line) - len(line.lstrip())] if stripped else None
            if stripped and len(cur_indent) <= len(rloop_indent):
                inside_rloop = False
                # fall through to normal handling for this line
            else:
                # Inside loop body. static -> dedent by 4 (loop removed);
                # dynamic -> indent by 4*(n_levels-1) (body sinks to innermost).
                if not stripped:
                    new_lines.append(line)
                    continue
                body_indent = (cur_indent + body_extra_indent) if dynamic else (cur_indent[4:] if len(cur_indent) >= 4 else cur_indent)  # noqa: B950
                # Rewrite a split-reconstruction alias (``r0_1 = r0_index``) into its
                # sub-node reconstruction so the flat node stays valid now r0_index is
                # gone. Must run BEFORE the scaffolding drop below.
                _lhs = stripped.split(" = ", 1)[0].strip() if " = " in stripped else None
                if _lhs in flat_recon:
                    new_lines.append(f"{body_indent}{_lhs} = {flat_recon[_lhs]}")
                    continue
                # Drop r0_index / r0_mask / roffset / rindex scaffolding and
                # the per-node mod/div decomposition (aranges replace them).
                if (
                    stripped.startswith(f"{prefix}index = ")
                    or stripped.startswith(f"{prefix}mask = ")
                    or stripped.startswith("roffset = ")
                    or stripped.startswith("rindex = ")
                    or _is_node_decomp(stripped)
                ):
                    continue
                # Rewrite the accumulate guard mask r0_mask -> combined mask.
                body = stripped
                if f"{prefix}mask & xmask" in body:
                    body = body.replace(
                        f"{prefix}mask & xmask",
                        f"({combined_mask}) & xmask",
                    )
                elif f"{prefix}mask" in body:
                    body = body.replace(prefix + "mask", f"({combined_mask})")
                new_lines.append(f"{body_indent}{body}")
                continue

        # Outside the r-loop now. Rewrite tl.sum / reduction lines: reshape-
        # collapse then resize. Matches ``X = tl.sum(ACC, D)[SLICE]``.
        if (not inside_rloop) and ("tl.sum(" in stripped or "triton_helpers." in stripped):
            import re as _re
            m = _re.match(
                r"^(\w+)\s*=\s*(tl\.sum|tl\.max|tl\.min|tl\.prod|tl\.xor_sum|triton_helpers\.\w+)\((\w+),\s*\d+\)(\[[^\]]*\])?\s*$",
                stripped,
            )
            if m:
                res, fn, acc, _slc = m.groups()
                rshp = f"{acc}_rc"
                new_lines.append(
                    f"{indent}{rshp} = tl.reshape({acc}, {collapsed_shape})"
                )
                new_lines.append(
                    f"{indent}{res} = {fn}({rshp}, {collapsed_dim}){post_resize}"
                )
                continue

        new_lines.append(line)

    kernel.body._lines = new_lines


# NPU does not require 32-aligned tile sizes; relax the upstream heuristic
# so that candidate 2D tilings with non-32-aligned y-dim are not rejected.
from torch._inductor.codegen.simd import CandidateTiling as _CandidateTiling

@staticmethod
def _npu_is_good_size(s):
    from torch._inductor.virtualized import V
    s = V.graph.sizevars.optimization_hint(s)
    return s >= 2

_CandidateTiling.is_good_size = _npu_is_good_size

# NPU does not support tl.float64/tl.int64 in compute. Patch the sympy printer
# so that Float constants and ToFloat casts use tl.float32 instead of tl.float64.
from torch._inductor.codegen.triton import TritonPrinter as _TritonPrinter

def _npu_print_Float(self, expr):
    return f"tl.full([], {expr}, tl.float32)"

def _npu_print_ToFloat(self, expr):
    assert len(expr.args) == 1
    from sympy.printing.precedence import PRECEDENCE
    s = self.parenthesize(expr.args[0], PRECEDENCE["Atom"] - 0.5)
    return f"{s}.to(tl.float32)"

_TritonPrinter._print_Float = _npu_print_Float
_TritonPrinter._print_ToFloat = _npu_print_ToFloat


# Tensor-dimension symbol kinds: any tensor actually indexed by a running kernel
# has every dim >= 1, so an expression built only from these is >= 1.
_DIM_SYMT = (SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.UNBACKED_INT)
_ONE = sympy.Integer(1)
# Inductor index strides use torch's own Max (torch.utils._sympy.functions.Max),
# which is a DISTINCT class from sympy.Max — `isinstance(node, sympy.Max)` and
# `expr.has(sympy.Max)` both miss it, even though the printer renders it as Max.
from torch.utils._sympy.functions import Max as _TorchMax


def _drop_size_clamp(expr):
    """Drop PyTorch's ``Max(1, dim)`` stride clamp from an index expression.

    Core clamps each contiguous stride dim by Max(1, dim); for a dynamic-shape conv
    the Triton printer (no ``max``) expands Max(1, ks0) into long repeated index
    arithmetic (dcgan load kernels). Max(1, e) is always just e here (every indexed
    dim >= 1), so dropping it yields the clean stride inductor already emits for the
    store side.
    """
    if not isinstance(expr, sympy.Expr) or not expr.has(_TorchMax):
        return expr

    def _is_clamp(node):
        return (
            isinstance(node, _TorchMax)
            and len(node.args) == 2
            and _ONE in node.args
        )

    def _replace(node):
        other = [a for a in node.args if a != _ONE]
        if len(other) != 1:
            return node
        e = other[0]
        syms = e.free_symbols
        # Only drop when e is provably >= 1: a non-empty set of dimension symbols.
        if syms and all(symbol_is_type(s, _DIM_SYMT) for s in syms):
            return e
        return node

    return expr.replace(_is_clamp, _replace)

# NPU: skip broadcast_to when all broadcast dims are 1 (degenerate/scalar kernel).
# Broadcasting a scalar to [1] creates a block-type value that can't be stored
# via a scalar pointer, causing "Value argument cannot be block type" errors.
_orig_codegen_broadcast_and_reshape = BlockPtrOptions.codegen_broadcast_and_reshape

def _npu_codegen_broadcast_and_reshape(self, value, initial_shape, final_shape, allow_implicit):
    if all(V.graph.sizevars.statically_known_equals(d, 1) for d in self.broadcast_shape):
        return value
    return _orig_codegen_broadcast_and_reshape(self, value, initial_shape, final_shape, allow_implicit)

BlockPtrOptions.codegen_broadcast_and_reshape = _npu_codegen_broadcast_and_reshape

# Without a user `other=`, inductor emits `other=0.0` for mask semantics, which on
# NPU forces a Vector pre-fill of the whole tile before MTE2, serializing movement
# behind compute. `care_padding=False` releases that when the region is unused.
npu_inject_care_padding = ncfg.inject_care_padding


# NPU r-axis cross-core split for OUTER reductions. When the free (x) axes are too
# few to fill all 40 cores, the default x-split idles most cores and each walks the
# whole reduction axis strided. Instead split into a "partial" kernel (slices the
# contiguous reduction axis across 40 cores, writes a [40, x] workspace) and a
# "combine" kernel (sums the 40 partials); both keep contiguous loads.
npu_rsplit_outer = ncfg.rsplit_outer


# NPU store-index clamp-stride refactor. Dynamic conv-output store strides Max(1,H)^k
# get expanded into a polynomial in D=H-1 (~6x repeated FloorDiv text); this finds the
# precomputed ks symbol the load side uses and subs D->sym-1 so (1+D)^k folds to sym^k.
# Runtime-exact. DEFAULT OFF: base-detection misfires on a layernorm-bwd transpose.
npu_refactor_clamp_stride = ncfg.refactor_clamp_stride



_CARE_PADDING_RE = None


def _rewrite_last_load_line(buffer, transform):
    """Apply ``transform`` to the last non-empty line of ``buffer`` (a plain string
    or a DelayReplaceLine wrapper, whose source is at ``.line``). ``transform``
    returns the new text or None to leave it. Returns True if it substituted."""
    lines = getattr(buffer, "_lines", None)
    if not lines:
        return False
    idx = len(lines) - 1
    while idx >= 0:
        candidate = lines[idx]
        if isinstance(candidate, str) and candidate.strip() == "":
            idx -= 1
            continue
        break
    if idx < 0:
        return False
    candidate = lines[idx]
    if isinstance(candidate, str):
        line = candidate

        def write_back(new):
            lines[idx] = new
    elif hasattr(candidate, "line") and isinstance(candidate.line, str):
        line = candidate.line

        def write_back(new):
            candidate.line = new
    else:
        return False
    new_line = transform(line)
    if new_line is None or new_line == line:
        return False
    write_back(new_line)
    return True


def _apply_to_first_buffer(transform, buffers):
    """super().load() may have written to self.loads, self.compute, or
    self.body depending on indexing/reduction context.  Try each until one
    line is rewritten."""
    for buf in buffers:
        if _rewrite_last_load_line(buf, transform):
            return


def _care_padding_transform(line):
    """Rewrite the trailing `, other=0.0)` of a `tl.load(...)` line to
    `, other=0.0, care_padding=False)`.  Returns None (leave as-is) if the line
    already has `care_padding` or lacks the inductor-default `other=0.0` — that
    protects user-supplied `other=` values."""
    if "tl.load(" not in line or "care_padding" in line:
        return None
    global _CARE_PADDING_RE
    if _CARE_PADDING_RE is None:
        import re as _re
        _CARE_PADDING_RE = _re.compile(r",\s*other=0\.0(\s*\))")
    new_line, n = _CARE_PADDING_RE.subn(
        lambda m: f", other=0.0, care_padding=False{m.group(1)}", line, count=1,
    )
    return new_line if n else None


# Upstream appends ``.to(tl.int1)`` to every bool-pointee ``tl.load`` (triton#2151).
# On Ascend the packed-i1 ``trunc i8 -> i1`` path overshoots masked tail addresses
# -> "ub address out of bounds". Rewriting to ``!= 0`` keeps the value bool but
# lowers via icmp instead of the packed-i1 trunc.
npu_rewrite_int1_cast_as_ne = ncfg.rewrite_int1_cast_as_ne


_INT1_CAST_RE = re.compile(r"(tl\.load\([^\n]*\))\.to\(tl\.int1\)")


def _int1_cast_transform(line):
    """Rewrite ``tl.load(...).to(tl.int1)`` to ``(tl.load(...) != 0)``."""
    if "tl.load(" not in line or ".to(tl.int1)" not in line:
        return None
    new_line, n = _INT1_CAST_RE.subn(r"(\1 != 0)", line, count=1)
    return new_line if n else None


def npu_triton_compute_type(dtype):
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name == "bool":
        triton_type_name = "int1"
    elif (
        triton_type_name in ("float16", "bfloat16")
        and ncfg.npu_triton.codegen_upcast_to_fp32
    ):
        triton_type_name = "float32"
    elif triton_type_name == "float8_e4m3fn":
        triton_type_name = "float8e4nv"
    elif triton_type_name == "float8_e5m2":
        triton_type_name = "float8e5"
    elif triton_type_name == "float8_e4m3fnuz":
        triton_type_name = "float8e4b8"
    elif triton_type_name == "float8_e5m2fnuz":
        triton_type_name = "float8e5b16"
    return f"tl.{triton_type_name}"
torch._inductor.codegen.triton.triton_compute_type = npu_triton_compute_type


class _FlatMapExpr:
    """Render-only adapter for collapse_rowmajor_xtree: quacks like a sympy expr
    (``free_symbols`` + ``str()``) but emits plain ``(F // d) % L`` the printer can
    lower without ModularIndexing/Mod plumbing."""
    __slots__ = ("free_symbols", "_text")

    def __init__(self, text, free_symbols):
        self._text = text
        self.free_symbols = set(free_symbols)

    def __str__(self):
        return self._text

    def __repr__(self):
        return f"_FlatMapExpr({self._text!r})"


# Reduction-guard elision: pad-lane abstract value (structural, SSA).
# Non-persistent reductions emit `_acc = tl.where(mask, combine(_acc, value), _acc)` only to
# hold pad lanes at the reduction identity (sum:0/prod:1/max:∓inf); tl.where is costly on NPU
# → drop when provably dead (value==identity). Each CSE var carries a pad-lane fact from
# load() propagated through the lattice: ("c",v) const | "fin" finite | "top" unknown (sound).

_PAD_TOP = "top"
_PAD_FIN = "fin"


def _pad_is_c(a):
    return isinstance(a, tuple) and len(a) == 2 and a[0] == "c"


def _pad_finite(a):
    if a == _PAD_FIN:
        return True
    if _pad_is_c(a):
        return math.isfinite(a[1])
    return False


def _pad_const(a):
    """The float constant of ``a`` if it is a known constant, else None."""
    return a[1] if _pad_is_c(a) else None


def _pad_add(a, b):
    if _pad_is_c(a) and _pad_is_c(b):
        return ("c", a[1] + b[1])
    if _pad_is_c(a) and not math.isfinite(a[1]) and _pad_finite(b):
        return a
    if _pad_is_c(b) and not math.isfinite(b[1]) and _pad_finite(a):
        return b
    if _pad_finite(a) and _pad_finite(b):
        return _PAD_FIN
    return _PAD_TOP


def _pad_sub(a, b):
    if _pad_is_c(a) and _pad_is_c(b):
        return ("c", a[1] - b[1])
    if _pad_is_c(a) and not math.isfinite(a[1]) and _pad_finite(b):
        return a
    if _pad_finite(a) and _pad_finite(b):
        return _PAD_FIN
    return _PAD_TOP


def _pad_mul(a, b):
    if _pad_is_c(a) and _pad_is_c(b):
        return ("c", a[1] * b[1])
    # 0 * finite == 0 (but 0 * inf is nan → only when the other side is finite)
    if a == ("c", 0.0) and _pad_finite(b):
        return ("c", 0.0)
    if b == ("c", 0.0) and _pad_finite(a):
        return ("c", 0.0)
    if a == ("c", 1.0):
        return b
    if b == ("c", 1.0):
        return a
    if _pad_finite(a) and _pad_finite(b):
        return _PAD_FIN
    return _PAD_TOP


def _pad_maxv(a, b):
    if _pad_is_c(a) and _pad_is_c(b):
        return ("c", max(a[1], b[1]))
    if _pad_finite(a) and _pad_finite(b):
        return _PAD_FIN
    return _PAD_TOP


def _pad_minv(a, b):
    if _pad_is_c(a) and _pad_is_c(b):
        return ("c", min(a[1], b[1]))
    if _pad_finite(a) and _pad_finite(b):
        return _PAD_FIN
    return _PAD_TOP


def _pad_expv(a):
    if _pad_is_c(a):
        try:
            return ("c", math.exp(a[1]))
        except OverflowError:
            return ("c", float("inf"))
    return _PAD_TOP  # exp(finite) may overflow → not provably finite


def _pad_join(a, b):
    """Join (least-upper-bound) of two pad facts — used for tl.where arms."""
    if _pad_is_c(a) and _pad_is_c(b) and a[1] == b[1]:
        return a
    if _pad_finite(a) and _pad_finite(b):
        return _PAD_FIN
    return _PAD_TOP


# How each op transforms its operands' pad facts.  Keyed by ops-handler
# method name; anything not listed falls to the conservative default
# (finite iff all operands finite, else top).  Constants/loads are seeded
# directly (constant(), load()) and never go through here.
def _pad_transform(name, arg_facts):
    if name == "constant":
        # constant(value, dtype): the exact literal is the pad value (it is
        # loop-invariant, same on valid and pad lanes).
        if arg_facts and _pad_is_c(arg_facts[0]):
            return arg_facts[0]
        return _PAD_FIN
    if name in ("add",):
        return _pad_add(arg_facts[0], arg_facts[1])
    if name in ("sub",):
        return _pad_sub(arg_facts[0], arg_facts[1])
    if name in ("mul",):
        return _pad_mul(arg_facts[0], arg_facts[1])
    if name in ("maximum",):
        return _pad_maxv(arg_facts[0], arg_facts[1])
    if name in ("minimum",):
        return _pad_minv(arg_facts[0], arg_facts[1])
    if name in ("exp",):
        return _pad_expv(arg_facts[0])
    if name in ("where",):
        # ops.where(cond, a, b): pad-lane value is the join of the two arms
        # (we do not model the condition).
        if len(arg_facts) >= 3:
            return _pad_join(arg_facts[1], arg_facts[2])
        return _PAD_TOP
    if name in (
        "to_dtype",
        "to_dtype_bitcast",
        "identity",
        "broadcast",
        "constant_to_device",
    ):
        # value-preserving w.r.t. the first operand's magnitude class
        return arg_facts[0] if arg_facts else _PAD_TOP
    if name in ("neg",):
        a = arg_facts[0]
        if _pad_is_c(a):
            return ("c", -a[1])
        return _PAD_FIN if _pad_finite(a) else _PAD_TOP
    if name in ("abs",):
        a = arg_facts[0]
        if _pad_is_c(a):
            return ("c", abs(a[1]))
        return _PAD_FIN if _pad_finite(a) else _PAD_TOP
    # Default: finite iff every operand finite, else unknown. Sound for any
    # finite->finite pointwise op; refusing to prove a constant only keeps a guard.
    if arg_facts and all(_pad_finite(f) for f in arg_facts):
        return _PAD_FIN
    return _PAD_TOP


# Reduction-load "other=" fill selection. A guard dead under no fill can die under
# another: softmax max/sum guards vanish if the load fills other=-inf (maximum(_acc,
# -inf)==_acc, exp(-inf-m)==0). Sound because other only fills masked lanes, which
# are either folded into a reduction (we pick only a fill proven on its identity) or
# re-masked by a store. Walk the FX forward cone, re-run the lattice per fill.

# Reduction identity as a float, per reduction_type (float reductions only).
_NPU_REDUCTION_IDENTITY = {
    "sum": 0.0,
    "xor_sum": 0.0,
    "any": 0.0,
    "prod": 1.0,
    "max": float("-inf"),
    "min": float("inf"),
}

# Candidate fills, in preference order.  0.0 is first so that when the default
# already elides (or nothing is provable) we change nothing.
_NPU_FILL_CANDIDATES = (0.0, float("-inf"), float("inf"), 1.0)


def _npu_body_of(kernel):
    node = getattr(kernel, "current_node", None)
    return getattr(node, "_body", None) if node is not None else None


def _npu_load_index_expr(kernel, load_node):
    """Resolved sympy index of an FX ``load`` node, via its get_index submodule
    and the LoopBody's ``indexing_exprs``.  None if not resolvable."""
    body = _npu_body_of(kernel)
    if body is None:
        return None
    try:
        gi = load_node.args[2]
        idx_name = gi.args[0]
    except Exception:
        return None
    return body.indexing_exprs.get(idx_name)


def _npu_load_touches_reduction(kernel, load_node):
    """True if the FX load's address depends on a LoopBody reduce var; None if
    the index cannot be resolved (unknown → treat as non-elidable)."""
    idx = _npu_load_index_expr(kernel, load_node)
    if idx is None:
        return None
    body = _npu_body_of(kernel)
    rvars = set(getattr(body, "reduce_vars", []) or [])
    return bool(idx.free_symbols & rvars)


def _npu_cone_value_fact(kernel, target_load, value_node, load_fill):
    """Evaluate the pad-lane fact of ``value_node`` (a reduction's value) when
    ``target_load`` is assigned ``load_fill`` and every OTHER reduction-axis
    load keeps its default (0.0) fill; reduction-invariant loads are FIN.

    Pure FX walk over the same lattice used at codegen; no side effects.
    """
    memo = {}

    def visit(node):
        key = id(node)
        if key in memo:
            return memo[key]
        if not hasattr(node, "op"):
            if isinstance(node, (int, float)) and not isinstance(node, bool):
                return ("c", float(node))
            return _PAD_TOP
        t = str(node.target)
        if t == "load":
            if node is target_load:
                f = load_fill
            else:
                touches = _npu_load_touches_reduction(kernel, node)
                if touches is None:
                    f = _PAD_TOP
                elif touches:
                    f = ("c", 0.0)  # default other=0.0 on a reduction-axis load
                else:
                    f = _PAD_FIN  # reduction-invariant load: finite real value
            memo[key] = f
            return f
        if t == "constant":
            v = node.args[1] if len(node.args) > 1 else None
            f = (
                ("c", float(v))
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                else _PAD_FIN
            )
            memo[key] = f
            return f
        if t in ("get_index", "index_expr", "load_seed", "randn", "rand"):
            memo[key] = _PAD_FIN
            return _PAD_FIN
        # Generic op: operand facts are node.args[1:] (skip the leading `ops`).
        arg_facts = []
        for a in node.args[1:]:
            if hasattr(a, "op"):
                arg_facts.append(visit(a))
            elif isinstance(a, (int, float)) and not isinstance(a, bool):
                arg_facts.append(("c", float(a)))
        try:
            f = _pad_transform(t, arg_facts)
        except Exception:
            f = _PAD_TOP
        memo[key] = f
        return f

    return visit(value_node)


def _npu_reductions_fed_by(load_node):
    """Every FX ``reduction`` node in the forward cone of ``load_node`` (the
    reductions whose accumulated value this load can influence)."""
    reds = []
    seen = set()
    stack = [load_node]
    while stack:
        n = stack.pop()
        for u in getattr(n, "users", ()):
            if id(u) in seen:
                continue
            seen.add(id(u))
            if str(getattr(u, "target", "")) == "reduction":
                reds.append(u)
            stack.append(u)
    return reds


def _npu_solve_reduction_fill(kernel, load_node):
    """Pick the ``other=`` fill for a reduction-axis load so that every
    reduction it feeds lands on its identity on pad lanes.

    Returns a float fill, or None to keep the default (0.0).  Returns None
    (conservative) whenever the choice is not unambiguously provable — an
    escape to a store other than through a satisfied reduction, a reduction
    type with no float identity, or no single fill satisfying all reductions.
    """
    reds = _npu_reductions_fed_by(load_node)
    if not reds:
        return None

    # Collect (reduction_node, value_node, identity) for each fed reduction.
    targets = []
    for r in reds:
        args = getattr(r, "args", ())
        if len(args) < 5:
            return None  # unexpected shape → bail
        rtype = args[3]
        ident = _NPU_REDUCTION_IDENTITY.get(rtype)
        if ident is None:
            return None  # welford/argmax/etc. — not handled
        targets.append((r, args[4], ident))

    # Find the fill that makes EVERY fed reduction land on its identity.  If
    # 0.0 already works everywhere, keep it (change nothing).
    for fill in _NPU_FILL_CANDIDATES:
        ok = True
        for _r, vnode, ident in targets:
            fact = _npu_cone_value_fact(kernel, load_node, vnode, ("c", fill))
            if fact != ("c", ident):
                ok = False
                break
        if ok:
            return fill
    return None


class NPUTritonCSEVariable(TritonCSEVariable):
    """A Triton CSE variable that also carries its reduction pad-lane fact.

    ``_npu_pad`` is the abstract value (see the lattice above) describing
    this variable's content on reduction pad lanes.  It is computed as the
    op is emitted, from the same operands upstream already passes to
    ``update_on_args`` — so it is exact SSA dataflow with no text parsing.
    ``load()`` seeds the fact for loads afterwards (the address/mask facts
    are only known there); everything else is derived here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default TOP: unknown until proven otherwise.  A bare/un-annotated
        # variable can never wrongly drop a guard.
        self._npu_pad = _PAD_TOP

    def update_on_args(self, name, args, kwargs):
        super().update_on_args(name, args, kwargs)
        # Propagate the pad-lane fact through this op.  Operands that are
        # NPU CSE vars carry their own fact; a bare python int/float operand
        # is an exact constant; anything else is unknown (TOP).
        arg_facts = []
        for a in args:
            if isinstance(a, NPUTritonCSEVariable):
                arg_facts.append(a._npu_pad)
            elif isinstance(a, (int, float)) and not isinstance(a, bool):
                arg_facts.append(("c", float(a)))
            elif isinstance(a, CSEVariable):
                arg_facts.append(_PAD_TOP)
            else:
                # sympy symbols, dtypes, strings (e.g. to_dtype's dtype arg)
                # are not value operands; ignore for the finite-join default
                # but keep constants that render as numbers.
                continue
        try:
            self._npu_pad = _pad_transform(name, arg_facts)
        except Exception:
            self._npu_pad = _PAD_TOP


class NPUTritonKernelOverrides(TritonKernelOverrides):
    @staticmethod
    def rsqrt(x):
        return f"tl.rsqrt({x})"

    @staticmethod
    def minimum(a, b):
        return f"tl.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"tl.maximum({a}, {b})"

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype=None, use_compute_types=True):
        if dtype == torch.float64:
            dtype = torch.float32
        elif dtype == torch.int64:
            dtype = torch.int32
        return TritonKernelOverrides.to_dtype(x, dtype, src_dtype, use_compute_types)

    @classmethod
    def constant(cls, value, dtype):
        if dtype == torch.float64:
            dtype = torch.float32
        elif dtype == torch.int64:
            dtype = torch.int32
        # When all tile dimensions are size 1 (degenerate kernel), use scalar
        # constant to avoid "Value argument cannot be block type if pointer
        # argument is not a block" error in tl.store.
        from torch._inductor.codegen.triton import TritonOverrides
        ndim = V.kernel.triton_tensor_ndim()
        if ndim == 0:
            return TritonOverrides.constant(value, dtype)
        # Check if all range tree dimensions have numel=1
        all_singleton = all(
            tree.numel == 1
            for tree in V.kernel.range_trees
            if not tree.is_reduction and tree.tensor_dim is not None
        )
        if all_singleton:
            return TritonOverrides.constant(value, dtype)
        return super().constant(value, dtype)

    @staticmethod
    def extract_slice(x, offsets, sizes, strides):
        """Register-level tile slice (bishengir CANN ``extract_slice``).

        First-class op form of the select-lane rewrite: ``x`` is a loaded tile
        that carries an extra local lane axis; ``extract_slice`` squeezes the
        chosen lane back out so downstream ops see the original tile shape.
        ``offsets``/``sizes``/``strides`` are per-tile-dim lists (Python lists
        or pre-rendered strings). Import is injected by gen_triton_ext_imports
        under NPU_SELECT_EXTRACT_SLICE.
        """
        def _fmt(v):
            return v if isinstance(v, str) else "[" + ", ".join(map(str, v)) + "]"

        return f"extract_slice({x}, {_fmt(offsets)}, {_fmt(sizes)}, {_fmt(strides)})"

    @staticmethod
    def cat_store(output_name, store_index, value, mask):
        """Direct masked store for the single-kernel concat (``NPUCatLoopKernel``).

        Emits ``tl.store(out_ptr + <flat store index>, value, <mask>)`` where
        ``mask`` is the ownership boolean VALUE built once at lowering time
        (``lo <= coord < hi``) -- the very same CSE var used to mask the load, so
        the ownership bound renders exactly once (rendering it twice desyncs the
        precomputed-size symbols and corrupts half the output). It is
        intersected with the ordinary range mask. No ``tl.where``: the caller
        loaded ``value`` under the same mask, so foreign-lane addresses are never
        issued and foreign lanes are never stored. Returns ``value`` so the
        recorded op has a result.
        """
        kernel = V.kernel
        indexing = kernel.indexing(store_index)

        own_mask = str(mask)
        base_mask = indexing.mask_str
        full_mask = f"{own_mask} & {base_mask}" if base_mask and base_mask != "None" else own_mask

        out_ptr = kernel.args.output(output_name)
        kernel.stores.writeline(
            f"tl.store({out_ptr} + ({indexing.index_str}), {value}, {full_mask})"
        )
        return value


    @staticmethod
    def index_select(src_name, weight_index, indirect_var, set_indirect, bound):
        """A5 CANN register row gather (``aten.embedding`` fast path).

        Emits ``extension.custom("__builtin_index_select", weight, idx, dim=0,
        ...)`` for the canonical embedding gather: a 2-D weight ``[V, H]``, gather
        along dim 0, a single indirect row axis, and a fully-resident contiguous
        inner ``H`` (stride 1). The op computes ``out[i][:] = weight[idx[i]][:]``
        as one register gather instead of ``H``-strided indirect ``tl.load``.

        Anything outside that shape (multiple indirect axes, tiled/strided H,
        non-unit inner stride, missing tile blocks) falls back to the plain
        indirect ``tl.load`` -- identical to the upstream gather -- so correctness
        never depends on the template applying. NOTE: the extension op only
        lowers on A5 (910_95); this path is gated by config.cann_index_select +
        is_a5() at the lowering layer and NEEDS A5 DEVICE VALIDATION (the mask /
        end-offset handling in particular).
        """
        from torch._inductor.utils import triton_type
        kernel = V.kernel

        def fallback_load(_reason):
            # weight_index already carries the indirect (TMP) symbol; the plain
            # load path (super().load -> indexing()) reproduces the upstream
            # H-strided gather exactly.
            return kernel.load(src_name, weight_index)

        try:
            tmp_syms = [s for s in weight_index.free_symbols
                        if symbol_is_type(s, SymT.TMP)]
            if len(tmp_syms) != 1:
                return fallback_load("not exactly one indirect axis")
            tmp = tmp_syms[0]
            coeffs = weight_index.as_coefficients_dict()
            row_stride = coeffs.get(tmp, None)          # == H (weight row stride)
            if not isinstance(row_stride, (int, sympy.Integer)) or int(row_stride) <= 0:
                return fallback_load("indirect stride not a positive int")
            H = int(row_stride)

            # Contiguous inner axes: every non-TMP symbol must be a real tiling
            # x-node, and the whole contiguous part must be a single stride-1
            # inner axis (the H lanes). Reject constant offsets / multi-axis H.
            inner_syms = [s for s in weight_index.free_symbols if s is not tmp]
            if len(inner_syms) != 1:
                return fallback_load(f"inner part not a single axis: {inner_syms}")
            h_sym = inner_syms[0]
            if int(coeffs.get(h_sym, 0)) != 1:
                return fallback_load("inner axis stride != 1")
            h_node = kernel.range_tree_nodes.get(h_sym)
            if h_node is None:
                return fallback_load("inner axis is not a range-tree node")
            # H must be fully resident in the tile (one block), so start_offset
            # into the row is 0 and end_offset spans the whole H lane count.
            if not V.graph.sizevars.statically_known_equals(h_node.length, H):
                return fallback_load("inner H axis is tiled (not fully resident)")
            h_block = f"real_block_{h_node.name}"

            # Row axis: the tiling axis the index tile depends on (its mask var).
            row_masks = [m for m in getattr(indirect_var, "mask_vars", ())
                         if m.endswith("mask") and m != "xmask"]
            if len(row_masks) != 1:
                return fallback_load(f"index tile spans !=1 masked axis: {row_masks}")
            row_name = row_masks[0][:-len("mask")]
            row_block = f"real_block_{row_name}"

            dtype = V.graph.get_dtype(src_name)
            tt = triton_type(dtype)
            w = kernel.args.input(src_name)
            cse, buf = kernel.cse, kernel.compute
            # The gather result broadcasts exactly like a plain load of
            # ``weight_index`` -- reuse that block shape so downstream (the store)
            # sees a consistently shaped tile. The reshape/full intermediates are
            # referenced only inside the opaque extension string, so their shape
            # only needs to satisfy the CSE-var assertion.
            out_shape = TritonSymbols.get_block_shape(weight_index)
            idx_shape = getattr(indirect_var, "shape", out_shape)

            idx1d = cse.generate(
                buf, f"tl.reshape({indirect_var}, ({row_block}, ))",
                dtype=indirect_var.dtype, shape=idx_shape)
            out_buf = cse.generate(
                buf, f"tl.full(({row_block}, {h_block}), 0, dtype={tt})",
                dtype=dtype, shape=out_shape)
            gathered = cse.generate(
                buf,
                f'extension.custom("__builtin_index_select", {w}, {idx1d}, '
                f'dim=0, bound={int(bound)}, '
                f'end_offset=({row_block}, {h_block}), start_offset=(0, 0), '
                f'src_stride=({H}, 1), out={out_buf})',
                dtype=dtype, shape=out_shape)
            return gathered
        except Exception as e:
            return fallback_load(f"exception {e!r}")


class NPUTritonKernel(TritonKernel):
    overrides = NPUTritonKernelOverrides  # type: ignore[assignment]

    def should_use_persistent_reduction(self) -> bool:
        # NPU does not support persistent reduction; always use the looped path
        # so that the accumulator (tl.full) is properly initialized.
        return False

    def create_cse_var(self, *args, **kwargs):
        # Use the NPU CSE variable so every value carries its reduction
        # pad-lane fact (see _pad_transform / NPUTritonCSEVariable), the
        # foundation of the reduction-guard elision in reduction().
        return NPUTritonCSEVariable(*args, **kwargs)

    @staticmethod
    def _reduction_identity_pad(reduction_type, src_dtype):
        """Pad-lane fact that the reduced value must match for the guard to be
        dead: the reduction's identity element as a float constant, or None if
        this reduction type is not elidable this way.

        sum/xor_sum/any → 0, prod → 1, max → -inf, min → +inf.  Only
        floating-point reductions are handled (integer identities like
        iinfo.min are representable but the value chain is modelled over floats;
        keeping them out is conservative, never wrong).
        """
        if not (src_dtype is not None and src_dtype.is_floating_point):
            return None
        if reduction_type in ("sum", "xor_sum", "any"):
            return ("c", 0.0)
        if reduction_type == "prod":
            return ("c", 1.0)
        if reduction_type == "max":
            return ("c", float("-inf"))
        if reduction_type == "min":
            return ("c", float("inf"))
        return None

    def reduction(self, dtype, src_dtype, reduction_type, value):
        """Elide the dead accumulator ``tl.where`` guard, structurally.

        Upstream emits ``_acc = tl.where(<r&x masks>, combine(_acc, value),
        _acc)`` per accumulation step.  That select only exists to pin the
        reduction PAD lanes at the identity; it is dead exactly when ``value``
        already equals the reduction identity on those lanes.  We know
        ``value``'s pad-lane fact structurally (``NPUTritonCSEVariable._npu_pad``,
        propagated from loads through every op with no text parsing), so when it
        matches the identity we drop the reduction masks for this one reduction
        — upstream's own ``where_cond`` then sees an empty condition and emits
        the plain ``_acc = combine(_acc, value)``.  No text is rewritten and the
        result is byte-identical on valid lanes.

        Safety: we only ever *drop* masks on a proven identity match; any
        unknown fact (``top``) keeps the guard.  A single-value reduction only
        (tuple/welford reductions keep their guards).
        """
        elide = False
        if (
            npu_elide_reduction_where
            and self.inside_reduction
            and not getattr(self, "persistent_reduction", False)
            and isinstance(value, NPUTritonCSEVariable)
        ):
            ident = self._reduction_identity_pad(reduction_type, src_dtype)
            if ident is not None and value._npu_pad == ident:
                elide = True

        if not elide:
            result = super().reduction(dtype, src_dtype, reduction_type, value)
            self._seed_reduction_result_pad(result, reduction_type, src_dtype)
            return result

        # Drop reduction masks for the duration of this reduction so the guard's
        # condition empties out and upstream emits no tl.where.
        self._npu_elide_reduction_masks = True
        try:
            result = super().reduction(dtype, src_dtype, reduction_type, value)
        finally:
            self._npu_elide_reduction_masks = False
        self._seed_reduction_result_pad(result, reduction_type, src_dtype)
        return result

    @staticmethod
    def _seed_reduction_result_pad(result, reduction_type, src_dtype):
        """Seed a reduction RESULT var's pad-lane fact.

        A float reduction (sum/prod/max/min/...) over real tensor data yields a
        FINITE scalar, and in a multi-pass reduction (e.g. softmax's max feeding
        the sum-exp pass) that result is broadcast and consumed on the next
        pass's pad lanes.  Without seeding it the framework leaves it TOP, which
        needlessly blocks elision of the downstream guard (``exp(load - max)``:
        with ``load`` = -inf and ``max`` = FIN the term is provably 0).  FIN is
        the sound fact — we never claim a specific constant, only finiteness.
        """
        if src_dtype is None or not src_dtype.is_floating_point:
            return
        if reduction_type not in ("sum", "xor_sum", "prod", "max", "min"):
            return
        if isinstance(result, NPUTritonCSEVariable):
            result._npu_pad = _PAD_FIN

    def filter_masks(self, mask_vars):
        super().filter_masks(mask_vars)
        # When reduction() proved the accumulator guard dead, discard range-tree masks so
        # the guard empties and upstream emits no tl.where. Both reduction and x/y/z masks
        # go: reduction because value is the identity on pad lanes, x/y/z because the acc on
        # invalid output rows is never stored (row independence). A semantic _load_mask is
        # appended AFTER filter_masks, so masked-region reductions keep their guard.
        if getattr(self, "_npu_elide_reduction_masks", False):
            for tree in self.range_trees:
                mask_vars.discard(f"{tree.prefix}mask")

    def load(self, name: str, index: sympy.Expr):
        restore_other = False
        saved_other = None
        if npu_elide_reduction_where and self._maybe_pick_reduction_load_fill():
            fill = self._pick_reduction_load_fill()
            if fill is not None:
                saved_other = self._load_other
                self._load_other = fill
                restore_other = True
        try:
            result_var = super().load(name, index)
            # Seed the pad-lane fact while the chosen ``other`` fill is still
            # live on self._load_other (the seeding reads it).
            self._maybe_record_select_lane_load(result_var, index)
            self._record_reduction_load_padinfo(result_var, index)
        finally:
            if restore_other:
                self._load_other = saved_other
        _bufs = [self.loads, self.compute, self.body]
        if npu_inject_care_padding:
            _apply_to_first_buffer(_care_padding_transform, _bufs)
        if npu_rewrite_int1_cast_as_ne:
            _apply_to_first_buffer(_int1_cast_transform, _bufs)
        return result_var

    def _maybe_pick_reduction_load_fill(self):
        """Cheap gate: only attempt fill selection for a reduction-axis load in
        a non-persistent reduction with no active semantic mask_loads override
        (``_load_other is None`` means upstream would default to 0.0)."""
        if not self.inside_reduction:
            return False
        if getattr(self, "persistent_reduction", False):
            return False
        if self._load_other is not None:
            # A torch-level masked_fill / index-put already pinned the fill; do
            # not second-guess it.
            return False
        return True

    def _pick_reduction_load_fill(self):
        """Return the non-default ``other`` fill for the CURRENT FX load node so
        that every reduction it feeds lands on its identity, or None to keep the
        default 0.0.  Delegates to the FX-cone solver; never raises."""
        load_node = getattr(V.interpreter, "current_node", None)
        if load_node is None or str(getattr(load_node, "target", "")) != "load":
            return None
        # Only reduction-axis loads can profit; an invariant load's pad lane is
        # already the finite real value and never needs a special fill.
        if _npu_load_touches_reduction(self, load_node) is not True:
            return None
        try:
            fill = _npu_solve_reduction_fill(self, load_node)
        except Exception:
            return None
        if fill is None or fill == 0.0:
            return None  # 0.0 is the default; nothing to change
        return fill


    def _record_reduction_load_padinfo(self, result_var, index: sympy.Expr):
        """Seed a loaded var's reduction pad-lane fact (``_npu_pad``).

        A reduction accumulator guard ``_acc = tl.where(mask, combine(_acc, v),
        _acc)`` is dead only if ``v`` equals the reduction identity on the pad
        lanes (``r_index >= r_numel``) that a non-mask-divisible tile exposes.
        Whether a given load is neutral on those lanes depends on the load's
        *address* and *mask* — known structurally HERE (sympy index +
        ``mask_vars``), NOT recoverable from ``other=`` text later.  This is the
        seed of the whole pad-lane dataflow: every other value derives its fact
        from these loads via ``NPUTritonCSEVariable.update_on_args``.

        Classification → seeded fact:
          * address does NOT depend on any reduction axis — the load is
            reduction-invariant, so its pad-lane value equals the (finite) valid
            value of the same row → ``FIN``.
          * address depends on a reduction axis AND every reduction mask is
            present in ``mask_vars`` — pad lanes are masked, so the pad value is
            exactly the load's ``other`` fill → ``("c", other)``.
          * address depends on a reduction axis but the mask does NOT cover it
            (a reshaped/flat index whose only mask is ``x``) — pad lanes read OUT
            OF BOUNDS, ``other`` does not apply → ``TOP``.  (The load()-time
            neutraliser upgrades this to the masked/``cov`` case when safe.)

        Only meaningful inside a non-persistent reduction; a no-op otherwise.
        """
        if not self.inside_reduction:
            return
        if getattr(self, "persistent_reduction", False):
            return
        if not isinstance(result_var, NPUTritonCSEVariable):
            return
        if not isinstance(index, sympy.Expr):
            return

        # Which reduction range-tree nodes does the address actually touch?
        addr_r_syms = set()
        for sym in index.free_symbols:
            node = self.range_tree_nodes.get(sym)
            if node is not None and node.root.is_reduction:
                addr_r_syms.add(sym)

        if not addr_r_syms:
            # Reduction-invariant load: same in-bounds address on pad lanes as on
            # valid lanes → finite (its concrete value is data-dependent).
            result_var._npu_pad = _PAD_FIN
            return

        # ``other`` fill for this load (default 0.0 when a mask is present).
        other = getattr(self, "_load_other", None)
        try:
            other_val = float(other) if other is not None else 0.0
        except Exception:
            other_val = None

        # Does the load's mask cover EVERY reduction range tree whose node the
        # address uses?  mask_vars is a set like {"r0_2mask", "r0_1mask",
        # "x0mask"}; a reduction mask token is "<prefix>mask".
        mask_vars = {str(m) for m in getattr(result_var, "mask_vars", ()) or ()}
        needed_prefixes = set()
        for sym in addr_r_syms:
            node = self.range_tree_nodes.get(sym)
            if node is not None:
                needed_prefixes.add(node.root.prefix)
        covered = all(f"{p}mask" in mask_vars for p in needed_prefixes)
        # Masked pad lanes → exact ``other`` fill; else (OOB / non-float other) unknown.
        result_var._npu_pad = ("c", other_val) if covered and other_val is not None else _PAD_TOP

    def _maybe_record_select_lane_load(self, result_var, index: sympy.Expr):
        """Structurally flag a stride-k select-lane load for the extract_slice
        rewrite, keyed by its result-var name.

        Detection is on the *sympy* index (not rendered text), at load() time
        where the index is still the flat pre-linearize form (e.g. ``2*x1`` for
        a ``x[...,0]`` select over a contiguous size-K inner axis). The criteria:

          * every free symbol is a free (non-reduction) range-tree node — a
            dynamic-shape stride carries a size symbol (``ks0``) that is NOT a
            range-tree node, so those bail here (correct: the lane isn't literal);
          * the index is affine with integer coefficients (no ModularIndexing /
            FloorDiv residue after removing the linear terms);
          * the minimum coefficient K is in {2,4,8} and there is NO unit-stride
            leaf — i.e. the innermost memory run is a strided gather of width K.

        The returned ``result_var`` names the exact load line codegen_body will
        rewrite, so no text classification is needed there. Gated on
        NPU_SELECT_EXTRACT_SLICE and only in linearize mode.
        """
        if not ncfg.select_extract_slice:
            return
        if not triton_codegen_linearize:
            return
        if self.inside_reduction:
            return
        var_name = getattr(result_var, "name", None)
        if not var_name or not isinstance(index, sympy.Expr):
            return

        leaves = {}
        for sym in index.free_symbols:
            node = self.range_tree_nodes.get(sym)
            if node is None or node.root.is_reduction:
                return  # non-node symbol (dynamic ks0) or reduction leaf → skip
            try:
                coeff = index.coeff(sym, 1)
            except Exception:
                return
            if not isinstance(coeff, (int, sympy.Integer)):
                return
            leaves[sym] = int(coeff)
        if not leaves:
            return
        # Affine: removing all linear terms must leave a pure constant (no sym).
        residue = index - sum(sympy.Integer(c) * s for s, c in leaves.items())
        if residue.free_symbols:
            return
        if any(c == 1 for c in leaves.values()):
            return  # a unit-stride leaf means the inner run is already contiguous
        k = min(leaves.values())
        if k not in (2, 4, 8):
            return
        self._npu_select_lane_loads[var_name] = k

    def index_to_str(self, index: sympy.Expr) -> str:
        # The index carries PyTorch's Max(1, dim) stride clamp from dynamic conv-output
        # layouts (torch.utils._sympy Max, not sympy.Max); the Triton printer has no max
        # and expands it into long repeated arithmetic. Strip it BEFORE rename_indexing
        # (ps0 → ks0): only pre-rename symbols carry the PRECOMPUTED_SIZE tag the strip
        # needs to prove dim >= 1. Mirror the upstream list path.
        if isinstance(index, list):
            return f"[{', '.join(map(self.index_to_str, index))}]"
        return self.kexpr(self.rename_indexing(_drop_size_clamp(index)))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._axis_split_subs: Dict[sympy.Symbol, sympy.Expr] = {}
        # r-axis cross-core split (OUTER reduction): when True, codegen_kernel
        # emits the "partial" form — program_id(0) selects a contiguous chunk of
        # the reduction axis (instead of the x-axis group dispatch), every core
        # walks the full x-tile loop, and the result is stored into a per-core
        # workspace row. Set by NPUTritonScheduling when the rsplit path triggers.
        self.npu_rsplit_partial = False
        # Workspace info filled in when npu_rsplit_partial is set:
        # (ws_inner_name, ws_offset_expr, x_total_numel_str, out_dtype).
        self.npu_rsplit_ws = None

        # extract_slice select-lane rewrite: result_var name -> lane stride k,
        # populated structurally at load() time (sympy index), consumed at
        # codegen_body time (see _maybe_rewrite_select_lane_load). Structural
        # detection replaces the old regex text classification.
        self._npu_select_lane_loads: Dict[str, int] = {}

        if triton_codegen_linearize:
            for tree in self.range_trees:
                if not hasattr(tree, 'tree_node_mapping'):
                    tree.tree_node_mapping = {}

    def _refactor_clamp_stride(self, index: sympy.Expr) -> sympy.Expr:
        """Re-fold an expanded ``(1+H)^k`` store stride onto a precomputed symbol.

        A store into this kernel's own conv-output-shaped buffer carries strides
        ``Max(1, dim)^k`` with ``dim = 1 + H`` and ``H`` a nested FloorDiv chain
        (e.g. ``((s-3)//4 - 3)//2``). Core drops the clamp and EXPANDS the power,
        so the per-axis stride prints as ``x1*(1+H) + x2*(1+2H+H^2) + ...`` and the
        long ``H`` text repeats ~6x in one index (see squeezenet1_1 backward). The
        matching *load* of the same ``dim`` registers ``Max(1, dim)`` as a
        precomputed size and prints the opaque ``ks`` arg instead.

        This pass restores that symmetry WITHOUT dropping the clamp:
          1. detect the shared compound base ``dim = 1 + H`` from the per-axis
             linear stride coefficients (the linear-stride axis has coeff dim);
          2. register ``Max(1, dim)`` via ``lookup_precomputed_size`` — the SAME
             key the load used, so it dedups to the load's ``ks`` symbol (or mints
             one arg if load didn't);
          3. substitute ``H -> sym - 1`` so ``(1+H)^k`` collapses to ``sym^k``,
             which prints as ``ks*ks`` exactly like the load side.

        Runtime-exact: ``sym = Max(1, 1+H) = 1+H`` since conv output dims are >= 1.
        No-op unless a compound ``(1+H)`` base with a FloorDiv chain is present.
        """
        if not isinstance(index, sympy.Expr) or not index.has(FloorDiv):
            return index
        sizevars = V.graph.sizevars
        var_ranges = self.var_ranges()
        iter_syms = [s for s in index.free_symbols if s in var_ranges]
        if not iter_syms:
            return index

        # Collect candidate bases: for each iteration var, factor its linear
        # coefficient; a compound base (carrying a FloorDiv) is the (1+H) dim.
        base = None
        for xi in iter_syms:
            coeff = index.coeff(xi, 1)
            if coeff == 0 or not coeff.has(FloorDiv):
                continue
            fac = sympy.factor(coeff)
            cand = fac.args[0] if isinstance(fac, sympy.Pow) else fac
            # Only accept a clean compound "1 + <FloorDiv chain>" form.
            if not (isinstance(cand, sympy.Add) and cand.has(FloorDiv)):
                continue
            if base is None:
                base = cand
            elif base != cand:
                # Mixed bases in one index — bail (keep upstream form).
                return index
        if base is None:
            return index

        h = sympy.expand(base - _ONE)  # H == dim - 1, the FloorDiv chain
        if not h.free_symbols or not index.has(h):
            return index

        # Register Max(1, dim) — identical key to the load's clamp so it dedups.
        try:
            sym = sizevars.lookup_precomputed_size(_TorchMax(_ONE, base))
        except Exception:
            return index
        if not isinstance(sym, sympy.Symbol):
            return index

        folded = sympy.expand(sympy_subs(index, {h: sym - _ONE}))
        if folded.has(FloorDiv):
            # Didn't fully collapse — don't emit a half-refactored index.
            return index
        # Numeric identity guard: sym == 1 + H.
        try:
            probe = {s: sympy.Integer(7) for s in (index.free_symbols | folded.free_symbols)}
            probe[sym] = sympy.Integer(1) + h.subs(probe)
            if sympy.simplify(index.subs(probe) - folded.subs(probe)) != 0:
                return index
        except Exception:
            return index
        return folded

    def prepare_indexing(self, index):
        index = super().prepare_indexing(index)
        if triton_codegen_linearize:
            index = self._maybe_split_fused_axes(index)
            index = self._maybe_split_strided_axis(index)
            index = self._simplify_compound_indexing(index)
            index = self._fold_dualview_reduction_index(index)
            # Re-apply precomputed replacements: _simplify_compound_indexing may
            # expand precomputed symbols (via inv_precomputed_replacements) and
            # leave expanded forms like FloorDiv(s1,4)*FloorDiv(s1,4) unreplaced.
            from torch._inductor.utils import sympy_subs
            index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if npu_refactor_clamp_stride:
            index = self._refactor_clamp_stride(index)
        index = self._fold_trivial_modular_indexing(index)
        return index

    def _simplify_compound_indexing(self, index: sympy.Expr) -> sympy.Expr:
        """Simplify ModularIndexing/FloorDiv on fused iteration variables.

        When a range tree has sub-nodes x0 (divisor=1, length=d) and x1
        (divisor=d), the fused variable x2 satisfies x2 = x0 + d*x1.
        So ModularIndexing(x2, d, m) = ModularIndexing(x1, 1, m) and
        FloorDiv(x2, d) = x1. This method performs these substitutions
        by looking up the tree structure directly.
        """
        if not index.has(ModularIndexing) and not index.has(FloorDiv):
            return index

        # Build a lookup: for each symbol that is a tree node,
        sizevars = V.graph.sizevars

        for tree in self.range_trees:
            if tree.is_reduction or tree.is_loop:
                continue
            all_nodes = list(tree.nodes.values())
            if len(all_nodes) < 2:
                continue
            # For each pair of nodes in the same tree, if node_a's divisor
            # matches the divisor used in ModularIndexing(node_b, divisor, m),
            # we can substitute.
            for node in all_nodes:
                # node.expr tells us how this node relates to the tree's index_sym
                # FloorDiv(xindex, d) means this node = xindex // d
                # ModularIndexing(xindex, d, l) means this node = (xindex // d) % l
                # The fused variable (xindex itself) has expr = FloorDiv(xindex, 1)
                pass

        # Direct approach: for each ModularIndexing(sym, d, m) in the index,
        # check if sym is a node in a tree that has another node at divisor d.
        # If so, that other node IS the result of sym // d (modulo its length).
        replacements: Dict[sympy.Expr, sympy.Expr] = {}

        for atom in index.atoms(ModularIndexing):
            base, divisor, modulus = atom.args
            if not isinstance(base, sympy.Symbol):
                # Handle compound base: ModularIndexing(c0*s0 + c1*s1 + ..., 1, m)
                # If all terms except one have coefficients divisible by m,
                # simplify to ModularIndexing(remaining_term, 1, m).
                if divisor == sympy.Integer(1) and base.is_Add:
                    terms = base.as_ordered_terms()
                    non_divisible = []
                    for term in terms:
                        coeff, _ = term.as_coeff_Mul()
                        if coeff == sympy.S.One:
                            # bare symbol, coeff=1, not divisible by m unless m==1
                            if modulus == sympy.Integer(1):
                                continue
                            non_divisible.append(term)
                        elif isinstance(coeff, sympy.Integer) and isinstance(modulus, sympy.Integer):
                            if int(coeff) % int(modulus) != 0:
                                non_divisible.append(term)
                        else:
                            non_divisible.append(term)
                    if len(non_divisible) == 1 and non_divisible[0] != base:
                        replacements[atom] = ModularIndexing(
                            non_divisible[0], sympy.Integer(1), modulus
                        )
                    elif len(non_divisible) > 1 and len(non_divisible) < len(terms):
                        remainder = sympy.Add(*non_divisible)
                        replacements[atom] = ModularIndexing(
                            remainder, sympy.Integer(1), modulus
                        )
                # Handle symbolic divisor: ModularIndexing(sum_of_terms, D, m)
                # If one term has coeff == D and the rest have max < D,
                # and the extracted var has range <= m, fold to that var.
                elif divisor != sympy.Integer(1) and base.is_Add:
                    var_ranges = self.var_ranges()
                    inv_precomputed = sizevars.inv_precomputed_replacements
                    divisor_expanded = sympy_subs(divisor, inv_precomputed) if inv_precomputed else divisor
                    terms = base.as_ordered_terms()
                    for i, term in enumerate(terms):
                        iter_vars_in_term = [v for v in term.free_symbols if v in var_ranges]
                        if len(iter_vars_in_term) != 1:
                            continue
                        var = iter_vars_in_term[0]
                        full_coeff = sympy.cancel(term / var)
                        coeff_expanded = sympy_subs(full_coeff, inv_precomputed) if inv_precomputed else full_coeff
                        try:
                            if not sizevars.statically_known_equals(coeff_expanded, divisor_expanded):
                                continue
                        except Exception:
                            continue
                        remainder_terms = terms[:i] + terms[i + 1:]
                        if not remainder_terms:
                            var_len = var_ranges[var]
                            try:
                                if sizevars.statically_known_leq(var_len, modulus):
                                    replacements[atom] = var
                            except Exception:
                                pass
                            break
                        remainder = sympy.Add(*remainder_terms)
                        subs = {}
                        ok = True
                        for v in remainder.free_symbols:
                            if v in var_ranges:
                                subs[v] = var_ranges[v] - 1
                            else:
                                ok = False
                                break
                        if not ok:
                            continue
                        try:
                            upper_remainder = sympy.expand(remainder.subs(subs))
                            upper_expanded = sympy_subs(upper_remainder, inv_precomputed) if inv_precomputed else upper_remainder
                            if not sizevars.statically_known_lt(upper_expanded, divisor_expanded):
                                continue
                        except Exception:
                            continue
                        var_len = var_ranges[var]
                        try:
                            if sizevars.statically_known_leq(var_len, modulus):
                                replacements[atom] = var
                                break
                        except Exception:
                            continue
                continue
            node = self.range_tree_nodes.get(base)
            if node is None:
                # base might be a tree's index_sym (e.g. xindex) which is not
                # in range_tree_nodes directly. Look up the tree by index_sym.
                tree = None
                for t in self.range_trees:
                    if t.index_sym() == base:
                        tree = t
                        break
                if tree is None or tree.is_reduction:
                    continue
                if divisor == sympy.Integer(1):
                    continue
                inv_precomputed = sizevars.inv_precomputed_replacements
                divisor_expanded = sympy_subs(divisor, inv_precomputed) if inv_precomputed else divisor
                for other in tree.nodes.values():
                    try:
                        other_div_expanded = sympy_subs(other.divisor, inv_precomputed) if inv_precomputed else other.divisor
                        if sizevars.statically_known_equals(other_div_expanded, divisor_expanded):
                            replacements[atom] = ModularIndexing(
                                other.symbol(), sympy.Integer(1), modulus
                            )
                            break
                    except Exception:
                        continue
                continue
            tree = node.root
            # We need: base // divisor. Look for a node in the same tree
            # whose divisor == divisor * node.divisor.
            # Skip when divisor==1: ModularIndexing(sym, 1, m) = sym % m,
            # which should be handled by split or fold, not by node lookup.
            if divisor == sympy.Integer(1):
                # Exception: `base` may be a FUSED node decomposed in tree_node_mapping
                # (x3 -> x5 + 64*x6); then ModularIndexing(x3, 1, m) drops terms whose coeff
                # divides m — for m==64 folds to x5. Without this the (x3 % 64) term stays a
                # non-linear NPU address (simplify skips divisor==1, second-pass split skips
                # mapped nodes). Gated by fold_fused_mod.
                if ncfg.fold_fused_mod:
                    mapped = tree.tree_node_mapping.get(node.name)
                    if mapped is not None and getattr(mapped, "is_Add", False) and isinstance(modulus, sympy.Integer):
                        non_divisible = []
                        ok = True
                        for term in mapped.as_ordered_terms():
                            coeff, _ = term.as_coeff_Mul()
                            if coeff == sympy.S.One:
                                non_divisible.append(term)
                            elif isinstance(coeff, sympy.Integer):
                                if int(coeff) % int(modulus) != 0:
                                    non_divisible.append(term)
                            else:
                                ok = False
                                break
                        if ok and len(non_divisible) < len(mapped.as_ordered_terms()):
                            if len(non_divisible) == 0:
                                replacements[atom] = sympy.Integer(0)
                            else:
                                remainder = sympy.Add(*non_divisible)
                                # If the surviving remainder is provably < modulus
                                # (e.g. x5 with range 64 <= modulus 64), the mod is
                                # the identity; else keep an explicit ModularIndexing
                                # on the smaller remainder.
                                var_ranges = self.var_ranges()
                                upper = remainder
                                try:
                                    subs = {v: var_ranges[v] - 1 for v in remainder.free_symbols if v in var_ranges}
                                    upper = sympy.expand(remainder.subs(subs)) if subs else remainder
                                except Exception:
                                    upper = None
                                replacements[atom] = remainder if upper is not None and sizevars.statically_known_lt(upper, modulus) else ModularIndexing(remainder, sympy.Integer(1), modulus)  # noqa: B950
                continue
            target_divisor = divisor * node.divisor
            inv_precomputed = sizevars.inv_precomputed_replacements
            target_expanded = sympy_subs(target_divisor, inv_precomputed) if inv_precomputed else target_divisor
            for other in tree.nodes.values():
                if other is node:
                    continue
                try:
                    other_div_expanded = sympy_subs(other.divisor, inv_precomputed) if inv_precomputed else other.divisor
                    if sizevars.statically_known_equals(other_div_expanded, target_expanded):
                        # other = base // divisor (within its length)
                        # So ModularIndexing(base, divisor, modulus) =
                        #    ModularIndexing(other, 1, modulus)
                        replacements[atom] = ModularIndexing(
                            other.symbol(), sympy.Integer(1), modulus
                        )
                        break
                except Exception:
                    continue

        for atom in index.atoms(FloorDiv):
            base, divisor = atom.args
            if not isinstance(base, sympy.Symbol):
                # Handle compound base: FloorDiv(c0*s0 + c1*s1 + ..., d)
                # Extract terms whose coefficients are divisible by d, leave
                # the rest under FloorDiv.
                # E.g. FloorDiv(384*x3 + x2, 64) → 6*x3 + FloorDiv(x2, 64)
                if isinstance(divisor, sympy.Integer) and base.is_Add:
                    terms = base.as_ordered_terms()
                    divisible_terms = []
                    remainder_terms = []
                    for term in terms:
                        coeff, rest = term.as_coeff_Mul()
                        if isinstance(coeff, sympy.Integer) and int(coeff) % int(divisor) == 0:
                            divisible_terms.append(sympy.Integer(int(coeff) // int(divisor)) * rest)
                        else:
                            remainder_terms.append(term)
                    if divisible_terms and remainder_terms:
                        extracted = sympy.Add(*divisible_terms)
                        remainder = sympy.Add(*remainder_terms)
                        replacements[atom] = extracted + FloorDiv(remainder, divisor)
                    elif divisible_terms and not remainder_terms:
                        replacements[atom] = sympy.Add(*divisible_terms)
                # Handle symbolic divisor: FloorDiv(sum_of_terms, D)
                # If one term has coeff == D and the rest have max < D,
                # fold to that variable.
                elif not isinstance(divisor, sympy.Integer) and base.is_Add:
                    var_ranges = self.var_ranges()
                    inv_precomputed = sizevars.inv_precomputed_replacements
                    divisor_expanded = sympy_subs(divisor, inv_precomputed) if inv_precomputed else divisor
                    terms = base.as_ordered_terms()
                    for i, term in enumerate(terms):
                        iter_vars_in_term = [v for v in term.free_symbols if v in var_ranges]
                        if len(iter_vars_in_term) != 1:
                            continue
                        var = iter_vars_in_term[0]
                        full_coeff = sympy.cancel(term / var)
                        coeff_expanded = sympy_subs(full_coeff, inv_precomputed) if inv_precomputed else full_coeff
                        try:
                            if not sizevars.statically_known_equals(coeff_expanded, divisor_expanded):
                                continue
                        except Exception:
                            continue
                        remainder_terms = terms[:i] + terms[i + 1:]
                        if not remainder_terms:
                            replacements[atom] = var
                            break
                        remainder = sympy.Add(*remainder_terms)
                        subs = {}
                        ok = True
                        for v in remainder.free_symbols:
                            if v in var_ranges:
                                subs[v] = var_ranges[v] - 1
                            else:
                                ok = False
                                break
                        if not ok:
                            continue
                        try:
                            upper_remainder = sympy.expand(remainder.subs(subs))
                            upper_expanded = sympy_subs(upper_remainder, inv_precomputed) if inv_precomputed else upper_remainder
                            if not sizevars.statically_known_lt(upper_expanded, divisor_expanded):
                                continue
                        except Exception:
                            continue
                        replacements[atom] = var
                        break
                continue
            node = self.range_tree_nodes.get(base)
            if node is None:
                # base might be a tree's index_sym (e.g. xindex)
                tree = None
                for t in self.range_trees:
                    if t.index_sym() == base:
                        tree = t
                        break
                if tree is None or tree.is_reduction:
                    continue
                inv_precomputed = sizevars.inv_precomputed_replacements
                divisor_expanded = sympy_subs(divisor, inv_precomputed) if inv_precomputed else divisor
                for other in tree.nodes.values():
                    try:
                        other_div_expanded = sympy_subs(other.divisor, inv_precomputed) if inv_precomputed else other.divisor
                        if sizevars.statically_known_equals(other_div_expanded, divisor_expanded):
                            replacements[atom] = other.symbol()
                            break
                    except Exception:
                        continue
                continue
            tree = node.root
            target_divisor = divisor * node.divisor
            inv_precomputed = sizevars.inv_precomputed_replacements
            target_expanded = sympy_subs(target_divisor, inv_precomputed) if inv_precomputed else target_divisor
            for other in tree.nodes.values():
                if other is node:
                    continue
                try:
                    other_div_expanded = sympy_subs(other.divisor, inv_precomputed) if inv_precomputed else other.divisor
                    if sizevars.statically_known_equals(other_div_expanded, target_expanded):
                        replacements[atom] = other.symbol()
                        break
                except Exception:
                    continue

        result = sympy_subs(index, replacements) if replacements else index

        # Second pass: split remaining FloorDiv(sym, c) / ModularIndexing(sym, 1, c)
        # by creating new sub-nodes in the tree (same logic as _maybe_split_fused_axes).
        split_subs: Dict[sympy.Expr, sympy.Expr] = {}
        split_candidates: Dict[sympy.Symbol, int] = {}
        for atom in result.atoms(FloorDiv):
            sym, c = atom.args
            if isinstance(sym, sympy.Symbol) and isinstance(c, sympy.Integer) and c > 1:
                split_candidates.setdefault(sym, int(c))
        for atom in result.atoms(ModularIndexing):
            base, divisor, modulus = atom.args
            if (
                isinstance(base, sympy.Symbol)
                and isinstance(divisor, sympy.Integer) and int(divisor) == 1
                and isinstance(modulus, sympy.Integer) and modulus > 1
            ):
                split_candidates.setdefault(base, int(modulus))

        for sym, c in split_candidates.items():
            if sym in split_subs:
                continue
            parent = self.range_tree_nodes.get(sym)
            if parent is None or parent.root.is_reduction:
                continue
            if parent.name in parent.root.tree_node_mapping:
                continue
            length = parent.length
            if not sizevars.statically_known_multiple_of(length, c):
                continue
            try:
                outer_len = length // c if isinstance(length, (int, sympy.Integer)) else FloorDiv(length, sympy.Integer(c))
            except Exception:
                continue
            if isinstance(outer_len, sympy.Integer) and int(outer_len) <= 1:
                continue
            tree = parent.root
            inner_entry = tree.lookup(parent.divisor, sympy.Integer(c))
            outer_entry = tree.lookup(parent.divisor * c, outer_len)
            outer_sym = outer_entry.symbol()
            inner_sym = inner_entry.symbol()
            tree.tree_node_mapping[parent.name] = outer_sym * c + inner_sym
            if hasattr(tree, 'var_tensor_dims'):
                tree.var_tensor_dims.pop(parent.name, None)
                next_dim = max(tree.var_tensor_dims.values(), default=-1) + 1
                tree.var_tensor_dims[outer_entry.name] = next_dim
                tree.var_tensor_dims[inner_entry.name] = next_dim + 1
            c_sym = sympy.Integer(c)
            split_subs[FloorDiv(sym, c_sym)] = outer_sym
            split_subs[ModularIndexing(sym, sympy.Integer(1), c_sym)] = inner_sym
            split_subs[sym] = outer_sym * c + inner_sym
            self._axis_split_subs[FloorDiv(sym, c_sym)] = outer_sym
            self._axis_split_subs[ModularIndexing(sym, sympy.Integer(1), c_sym)] = inner_sym
            self._axis_split_subs[sym] = outer_sym * c + inner_sym

        if split_subs:
            result = sympy_subs(result, split_subs)

        # Final simplification
        try:
            var_ranges = self.var_ranges()
            return sizevars.simplify_with_ranges(result, var_ranges)
        except Exception:
            return result

    def _fold_dualview_reduction_index(self, index: sympy.Expr) -> sympy.Expr:
        """Linearize a dual-view reduction sub-axis reconstruction in an address.

        A reduction range-tree may view its inner flat block two ways: as a flat
        node ``r0_1 = r0_index`` AND as a contiguous sub-axis chain
        ``r0_3 = r0_index % ks1``, ``r0_4 = (r0_index // ks1) % ks1`` that tiles
        the SAME block. Loads that index via the chain carry the reconstruction
        ``r0_3 + ks1*r0_4`` in their address. That sum is algebraically the
        linear inner index ``r0_1`` (since ``r0_index < ks1^2`` makes the outer
        ``% ks1`` an identity), but written with the chain symbols bishengir
        cannot see it as contiguous and degrades the load to a scalar gather.

        Fold the chain reconstruction back to the inner node symbol in the sympy
        index — order-independent (the chain terms may be interspersed with other
        axes, e.g. ``r0_3 + ks0*x0 + ks1*r0_4 + ...``), running before the address
        is rendered. Only fires for a reduction tree whose aliased nodes form a
        complete divisor chain tiling exactly the inner block; genuine
        independent reduction axes (span exceeds the inner block) never match.
        Gated by NPU_FOLD_DUALVIEW_RNODE (default on); mirrors the x-tree
        treatment in _fold_dual_decomp.
        """
        if not ncfg.fold_dualview_rnode:
            return index
        if not getattr(self, "inside_reduction", False):
            return index
        sizevars = V.graph.sizevars
        free_syms = index.free_symbols
        # Reduction range symbols carry sympy assumptions (integer/nonnegative),
        # so a freshly built sympy.Symbol(name) is a DIFFERENT object that never
        # compares equal / hashes into free_syms. Match by NAME and reuse the
        # actual symbol object from the index so the algebraic fold cancels.
        free_by_name = {s.name: s for s in free_syms}

        def _hint(e):
            return int(sizevars.optimization_hint(e))

        for tree in self.range_trees:
            if not tree.is_reduction:
                continue
            nodes = list(tree.nodes.values())
            if len(nodes) < 3:
                # Need an inner flat node + at least a 2-node alias chain.
                continue

            nodes_sorted = sorted(nodes, key=lambda n: _hint(n.divisor))
            # Inner flat node = divisor-1 node with the LARGEST length (the flat
            # view of the whole inner block). A naive min-divisor pick is
            # ambiguous when the alias chain's low node ALSO has divisor 1.
            div1 = [n for n in nodes_sorted
                    if isinstance(n.divisor, (int, sympy.Integer)) and int(n.divisor) == 1]
            if not div1:
                continue
            inner = max(div1, key=lambda n: _hint(n.length))
            inner_span = _hint(inner.length)

            # Candidate alias nodes: present in the index (matched by NAME),
            # contained in the inner block (divisor*length <= inner_span),
            # excluding the inner node.
            candidates = [
                n for n in nodes_sorted
                if n is not inner
                and n.name in free_by_name
                and _hint(n.divisor) * _hint(n.length) <= inner_span
            ]
            if len(candidates) < 2:
                continue

            # The candidates must form a contiguous divisor chain tiling exactly
            # the inner block: divisors 1, L0, L0*L1, ... and product == inner.
            chain = sorted(candidates, key=lambda n: _hint(n.divisor))
            if not (isinstance(chain[0].divisor, (int, sympy.Integer))
                    and int(chain[0].divisor) == 1):
                continue
            expected = sympy.Integer(1)
            chain_ok = True
            for o in chain:
                if not sizevars.statically_known_equals(o.divisor, expected) and _hint(o.divisor) != _hint(expected):
                    chain_ok = False
                    break
                expected = o.divisor * o.length
            if not chain_ok:
                continue
            if not sizevars.statically_known_equals(expected, inner.length) and _hint(expected) != inner_span:
                continue

            # Reconstruction Σ alias_i * divisor_i == inner flat index. Build the
            # term from the index's OWN symbol object (assumptions intact) and the
            # node's stride/divisor as it appears in the address. The divisor may
            # itself be symbolic (e.g. the dynamic width s82), so multiply by the
            # node.divisor expression rather than its hint.
            terms = []
            for o in chain:
                sym = free_by_name[o.name]
                if isinstance(o.divisor, (int, sympy.Integer)) and int(o.divisor) == 1:
                    terms.append(sym)
                else:
                    terms.append(sym * o.divisor)
            chain_flat = sympy.Add(*terms)
            inner_sym = free_by_name.get(inner.name, sympy.Symbol(inner.name))

            # Order-independent additive fold: index - chain_flat + inner_sym.
            # Only commit if it actually removes the chain symbols (i.e. the full
            # reconstruction was present), so partial matches are left alone.
            folded = sympy.expand(index - chain_flat + inner_sym)
            chain_names = {o.name for o in chain}
            residual = {s for s in folded.free_symbols if s.name in chain_names}
            if residual:
                continue  # reconstruction not fully present in this index
            index = folded
            free_syms = index.free_symbols
            free_by_name = {s.name: s for s in free_syms}

        return index

    def _fold_trivial_modular_indexing(self, index: sympy.Expr) -> sympy.Expr:
        """Drop ``ModularIndexing(base, 1, modulus)`` when the substitution
        ``v -> upper(v) - 1`` for every iteration var in ``base`` produces a
        value strictly less than ``modulus`` — i.e. the wrap is statically a
        no-op even when symbolic shapes (e.g. ``s0``) defeat upstream
        ``simplify_with_ranges`` axioms. Each var range is treated as
        non-negative, which matches how Inductor names iteration vars.

        Also folds ``FloorDiv(base, divisor)`` to 0 when the max value of
        ``base`` is strictly less than ``divisor``.
        """
        if not isinstance(index, sympy.Basic):
            return index
        if not index.has(ModularIndexing) and not index.has(FloorDiv):
            return index
        var_ranges = self.var_ranges()
        if not var_ranges:
            return index
        sizevars = V.graph.sizevars
        replacements: Dict[sympy.Expr, sympy.Expr] = {}
        for atom in index.atoms(ModularIndexing):
            base, divisor, modulus = atom.args
            if divisor != 1:
                continue
            subs = {}
            ok = True
            for v in base.free_symbols:
                if v not in var_ranges:
                    ok = False
                    break
                subs[v] = var_ranges[v] - 1
            if not ok:
                continue
            try:
                upper = sympy.expand(base.subs(subs))
            except Exception:
                continue
            try:
                if sizevars.statically_known_lt(upper, modulus):
                    replacements[atom] = base
            except Exception:
                continue
        for atom in index.atoms(FloorDiv):
            base, divisor = atom.args
            if not isinstance(divisor, sympy.Integer) or divisor <= 1:
                continue
            subs = {}
            ok = True
            for v in base.free_symbols:
                if v not in var_ranges:
                    ok = False
                    break
                subs[v] = var_ranges[v] - 1
            if not ok:
                continue
            try:
                upper = sympy.expand(base.subs(subs))
            except Exception:
                continue
            try:
                if sizevars.statically_known_lt(upper, divisor):
                    replacements[atom] = sympy.Integer(0)
            except Exception:
                continue
        if not replacements:
            return index
        return sympy_subs(index, replacements)

    def _maybe_split_fused_axes(self, index: sympy.Expr) -> sympy.Expr:
        """Split a fused iteration axis into outer/inner sub-axes when an index
        accesses it via FloorDiv(x, c) or ModularIndexing(x, 1, c).

        Linearize-mode codegen otherwise emits `tl.load(... + (x // c) ... + (x % c))`,
        which the NPU backend lowers to scalar / indirect addressing. Splitting the
        node lets the index become a pure linear expression of independent axes.

        We trigger on the union of the two forms (rather than the intersection)
        because a single fused axis can show up in separate index expressions —
        e.g. select_scatter backward emits `x0 % c` in the load index and
        `x0 // c` in a separate `index_expr` for the equality predicate. Seeing
        either form is enough evidence that `x` is a flattened (inner * c +
        outer * 1)-style axis; once we register both substitutions in
        ``_axis_split_subs`` the other index that comes through later will be
        rewritten too.
        """
        if self._axis_split_subs:
            index = sympy_subs(index, self._axis_split_subs)

        candidates: Dict[sympy.Symbol, sympy.Expr] = {}
        for atom in index.atoms(FloorDiv):
            sym, c = atom.args
            if not isinstance(sym, sympy.Symbol):
                continue
            if isinstance(c, sympy.Integer) and c > 1:
                candidates.setdefault(sym, sympy.Integer(c))
            elif not c.is_number and c.is_positive:
                # Symbolic split factor (e.g. FloorDiv(y0, s98) on a fused axis
                # of length 2*s98). Splitting still produces a pure-affine index
                # provided c divides the axis length, which is checked below.
                candidates.setdefault(sym, c)
        for atom in index.atoms(ModularIndexing):
            base, divisor, modulus = atom.args
            if not (
                isinstance(base, sympy.Symbol)
                and isinstance(divisor, sympy.Integer) and int(divisor) == 1
            ):
                continue
            if isinstance(modulus, sympy.Integer) and modulus > 1:
                candidates.setdefault(base, sympy.Integer(modulus))
            elif not modulus.is_number and modulus.is_positive:
                # Symbolic modulus, e.g. ModularIndexing(y0, 1, s98): fused axis y0
                # (length 2*s98) carries an expand/broadcast sub-axis. Split y0 ->
                # outer(=2)*s98 + inner(=s98) so y0 % s98 folds to `inner` and the index
                # becomes affine (matches the un-merged separate-axis layout).
                candidates.setdefault(base, modulus)

        if not candidates:
            return index

        sizevars = V.graph.sizevars
        seen_syms = set()
        new_subs = {}
        for sym, c in sorted(candidates.items(), key=lambda p: (str(p[0]), str(p[1]))):
            if sym in seen_syms:
                continue
            parent = self.range_tree_nodes.get(sym)
            if parent is None:
                continue
            # Reduction axes are normally left unsplit (r-loop drives tiling). But a
            # fused reduction axis carrying a ModularIndexing (expand/broadcast read, e.g.
            # sum over p1=[0,s27*s33) with `p1 % s27`) leaves a non-linear `r % c` the NPU
            # lowers to a scalar gather. Split p1 into inner(c)/outer to fold `p1 % c` to
            # the inner axis → affine load. Gated; historical path unaffected when off.
            if parent.root.is_reduction and not ncfg.split_reduction_modulo:
                continue

            tree = parent.root

            # Root-modulo redirect: when `sym` is the fused ROOT the modulo arrives as
            # ModularIndexing(root, 1, c) though the tile is driven by the stride-1 LEAF
            # (AFN+ BN: root x2 = x0 + 58500*x1, gather x2 % 1500); splitting the root mints
            # colliding axes. Since every leaf divisor is a multiple of c, root % c == leaf
            # % c; redirect the split onto the leaf and fold root's modulo onto it. Gated.
            split_sym = sym
            split_node = parent
            root_mod_redirect = None  # (root_sym, inner) to also fold root % c
            _redirect_on = ncfg.split_root_modulo
            if (
                _redirect_on
                and isinstance(parent.divisor, (int, sympy.Integer)) and int(parent.divisor) == 1
                and isinstance(c, sympy.Integer)
                and isinstance(parent.length, (int, sympy.Integer))
            ):
                leaf = None
                for n in tree.nodes.values():
                    if n is parent or n.name in tree.tree_node_mapping:
                        continue
                    if not (isinstance(n.divisor, (int, sympy.Integer)) and int(n.divisor) == 1):
                        continue
                    if not isinstance(n.length, (int, sympy.Integer)):
                        continue
                    if int(n.length) >= int(parent.length) or int(n.length) <= int(c):
                        continue
                    if int(n.length) % int(c) != 0:
                        continue
                    # Every remaining leaf's stride must be a multiple of c so
                    # that root % c == leaf % c (the dropped terms vanish mod c).
                    others_ok = True
                    for m in tree.nodes.values():
                        if m is parent or m is n or m.name in tree.tree_node_mapping:
                            continue
                        if not (isinstance(m.divisor, (int, sympy.Integer))
                                and int(m.divisor) % int(c) == 0):
                            others_ok = False
                            break
                    if others_ok and (leaf is None or int(n.length) < int(leaf.length)):
                        leaf = n
                if leaf is not None:
                    split_sym = leaf.symbol()
                    split_node = leaf
                    root_mod_redirect = sym

            if split_sym in seen_syms:
                continue
            if split_node.name in tree.tree_node_mapping:
                continue
            length = split_node.length
            if not sizevars.statically_known_multiple_of(length, c):
                continue
            try:
                if isinstance(length, (int, sympy.Integer)) and isinstance(c, sympy.Integer):
                    outer_len = length // c
                else:
                    # c divides length exactly (guarded above); cancel to get a
                    # clean factor rather than an unevaluated FloorDiv residue.
                    outer_len = sympy.cancel(sympy.sympify(length) / sympy.sympify(c))
            except Exception:
                continue
            if isinstance(outer_len, sympy.Integer) and int(outer_len) <= 1:
                continue

            inner_entry = tree.lookup(split_node.divisor, c)
            outer_entry = tree.lookup(split_node.divisor * c, outer_len)
            outer_sym = outer_entry.symbol()
            inner_sym = inner_entry.symbol()

            tree.tree_node_mapping[split_node.name] = outer_sym * c + inner_sym

            if hasattr(tree, 'var_tensor_dims'):
                tree.var_tensor_dims.pop(split_node.name, None)
                next_dim = max(tree.var_tensor_dims.values(), default=-1) + 1
                tree.var_tensor_dims[outer_entry.name] = next_dim
                tree.var_tensor_dims[inner_entry.name] = next_dim + 1

            new_subs[FloorDiv(split_sym, c)] = outer_sym
            new_subs[ModularIndexing(split_sym, sympy.Integer(1), c)] = inner_sym
            new_subs[split_sym] = outer_sym * c + inner_sym
            seen_syms.add(split_sym)

            if parent.root.is_reduction:
                # A split reduction axis becomes an inner/outer nested loop, but the flat
                # parent stays live in loads emitted BEFORE the split (`768*r0_1`, no
                # FloorDiv/Mod → never picked up _axis_split_subs). Record its recon (flat =
                # inner + c*outer) so codegen_body rewrites `r0_1 = r0_index` → `r0_2 +
                # ks0*r0_3` and flips _flat_subs_active on; rename_indexing maps `c` (s27).
                flat_subs = getattr(self, "_npu_flat_rnode_subs", None)
                if flat_subs is None:
                    flat_subs = {}
                    self._npu_flat_rnode_subs = flat_subs
                recon = inner_sym + outer_sym * c
                flat_subs[split_node.name] = self.kexpr(self.rename_indexing(recon))
                # Tag as a split-created RECONSTRUCTION alias (flat = inner + c*outer over
                # sub-nodes promotion DOES emit), distinct from a dual-VIEW alias (flat =
                # r0_index scaffolding promotion DELETES). The rtree gate bails on any
                # mapping; this tag lets it allow ours and reconstruct the flat node from
                # the promoted aranges instead of nested scalar loops.
                recon_nodes = getattr(tree, "_npu_split_recon_nodes", None)
                if recon_nodes is None:
                    recon_nodes = set()
                    tree._npu_split_recon_nodes = recon_nodes
                recon_nodes.add(split_node.name)

            if root_mod_redirect is not None:
                # root % c == leaf % c == inner sub-axis (dropped strides ≡ 0 mod c).
                new_subs[ModularIndexing(root_mod_redirect, sympy.Integer(1), c)] = inner_sym
                seen_syms.add(root_mod_redirect)

        if not new_subs:
            return index

        self._axis_split_subs.update(new_subs)
        index = sympy_subs(index, new_subs)
        # Re-run range-based simplification so that ``(outer*c + inner) % c``
        # collapses to ``inner`` and ``(outer*c + inner) // c`` to ``outer``
        # using the new sub-axes' var_ranges.
        try:
            index = sizevars.simplify_with_ranges(index, self.var_ranges())
        except Exception:
            pass
        return index

    def _maybe_split_strided_axis(self, index: sympy.Expr) -> sympy.Expr:
        """Split a collapsed pointwise axis carrying a strided (coeff>1) load.

        When a pointwise op's iteration domain fully collapses to a single flat
        x-axis (all input/output strides are concrete and uniformly proportional,
        e.g. ``out[...,1::2]`` over a contiguous buffer folds ``128*x1 + 2*x0``
        into ``2*x0``), the resulting kernel does ONE wide stride-``coeff`` gather
        over the whole axis inside a single XBLOCK loop tile. The NPU backend
        cannot vectorise a wide strided gather — it lowers to scalar/indirect
        addressing and runs ~7x slower than necessary (measured: 681us vs 100us
        on the qkv-rmsnorm-rope odd-lane slice, S=1024).

        The dynamic-shape variant of the same graph never collapses, because a
        symbolic reshape divisor keeps the strides non-proportional, so it keeps
        the strided gather bounded to a small inner axis and parallelises the
        rest across cores. This method reproduces that shape for static shapes:
        when the (single, unsplit) free x-leaf appears in ``index`` with a
        constant coefficient > 1, split it into ``outer*INNER + inner`` so the
        stride-``coeff`` access is confined to the INNER-wide innermost axis and
        the OUTER axis tiles across the 40 cores. INNER is the largest power-of-two
        divisor of the axis length that is <= 64 (matches the dynamic kernel's
        inner-tile width and stays within UB on bf16/fp32 tiles).

        Numerically a pure reindex (identity on the iteration domain) — only the
        loop nest shape changes. Gated by NPU_SPLIT_STRIDED_POINTWISE (default ON).
        """
        if not ncfg.split_strided_pointwise:
            return index
        # Pure pointwise only: a reduction kernel's tiling is shaped by the
        # r-axis, and splitting the x-axis there is handled elsewhere.
        if getattr(self, "inside_reduction", False):
            return index

        if self._axis_split_subs:
            index = sympy_subs(index, self._axis_split_subs)

        sizevars = V.graph.sizevars

        # Identify the free (non-reduction) x-leaf nodes that are NOT already
        # split. The collapse case we target has exactly one such leaf spanning
        # the whole xnumel; if there are already >= 2 axes the kernel is already
        # multi-axis and the strided gather is already bounded.
        free_leaves = []
        for sym in index.free_symbols:
            node = self.range_tree_nodes.get(sym)
            if node is None or node.root.is_reduction:
                continue
            if node.name in node.root.tree_node_mapping:
                continue
            free_leaves.append((sym, node))
        if len(free_leaves) != 1:
            return index

        sym, parent = free_leaves[0]

        # The load that motivates the split: index must be affine in ``sym`` with
        # an integer coefficient > 1. (A contiguous store has coeff 1 and won't
        # trigger; once the split is registered the store is rewritten via subs.)
        try:
            coeff = index.coeff(sym, 1)
        except Exception:
            return index
        if not (isinstance(coeff, (int, sympy.Integer)) and int(coeff) > 1):
            return index
        # Affine check: removing the linear term must leave sym out of the index.
        if (index - coeff * sym).has(sym):
            return index

        length = parent.length
        # Pick INNER = largest power-of-two divisor of length that is <= 64 and
        # leaves outer length > 1 (so the split is meaningful).
        inner = None
        for cand in (64, 32, 16, 8):
            if not sizevars.statically_known_multiple_of(length, cand):
                continue
            try:
                outer_len = (
                    length // cand
                    if isinstance(length, (int, sympy.Integer))
                    else sympy.cancel(sympy.sympify(length) / cand)
                )
            except Exception:
                continue
            if isinstance(outer_len, sympy.Integer) and int(outer_len) <= 1:
                continue
            inner = cand
            break
        if inner is None:
            return index

        c = sympy.Integer(inner)
        outer_len = (
            length // inner
            if isinstance(length, (int, sympy.Integer))
            else sympy.cancel(sympy.sympify(length) / c)
        )

        tree = parent.root
        inner_entry = tree.lookup(parent.divisor, c)
        outer_entry = tree.lookup(parent.divisor * c, outer_len)
        outer_sym = outer_entry.symbol()
        inner_sym = inner_entry.symbol()

        tree.tree_node_mapping[parent.name] = outer_sym * c + inner_sym

        if hasattr(tree, 'var_tensor_dims'):
            tree.var_tensor_dims.pop(parent.name, None)
            next_dim = max(tree.var_tensor_dims.values(), default=-1) + 1
            # OUTER tiles across cores (outermost dim), INNER stays innermost so
            # the contiguous store keeps unit stride on the last dim.
            tree.var_tensor_dims[outer_entry.name] = next_dim
            tree.var_tensor_dims[inner_entry.name] = next_dim + 1

        new_subs = {
            FloorDiv(sym, c): outer_sym,
            ModularIndexing(sym, sympy.Integer(1), c): inner_sym,
            sym: outer_sym * c + inner_sym,
        }
        self._axis_split_subs.update(new_subs)
        index = sympy_subs(index, new_subs)
        try:
            index = sizevars.simplify_with_ranges(index, self.var_ranges())
        except Exception:
            pass
        return index

    def initialize_range_tree(self, pid_cache):
        """Override to add tree_node_mapping for linearize mode."""
        super().initialize_range_tree(pid_cache)
        if triton_codegen_linearize:
            for tree in self.range_trees:
                if not hasattr(tree, 'tree_node_mapping'):
                    tree.tree_node_mapping = {}

    def gen_triton_ext_imports(self):
        imports = IndentedBuffer()
        imports.splice(
            """
            from torch_npu._inductor.triton_experimental import npu_triton_heuristics
            from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice, math as tl_math
            """
        )
        # Register-level slice op used by the stride-k select-lane load rewrite
        # (_maybe_extract_slice_lane_load). Only imported when the feature is on
        # so the vast majority of kernels keep their import block unchanged.
        if ncfg.select_extract_slice:
            imports.splice(
                """
                from triton.language.extra.cann.extension import extract_slice
                """
            )
        # A5 CANN register gather (embedding __builtin_index_select). The
        # ``extension`` module alias backs ops.index_select's codegen. Imported
        # only when the feature is enabled AND we are on A5 (the ops only lower
        # there), so non-A5 kernels keep their import block unchanged.
        if ncfg.cann_index_select and device_props.is_a5():
            imports.splice(
                """
                import triton.language.extra.cann.extension as extension
                """
            )
        return imports.getvalue()

    def triton_tensor_ndim(self):
        if triton_codegen_linearize and getattr(self, '_linearize_applied', False):
            ndim = 0
            for tree in self.range_trees:
                if tree.tensor_dim is not None:
                    if not tree.is_reduction:
                        _scalar_odo = getattr(tree, '_npu_scalar_odometer', None) or set()
                        for node in tree.nodes.values():
                            if (node.name not in getattr(tree, 'tree_node_mapping', {})
                                    and node.name not in _scalar_odo):
                                ndim += 1
                    else:
                        # Promoted reduction trees occupy one slot PER surviving
                        # free sub-node (real-block multi-tile), mirroring the
                        # x-tree count above; non-promoted r-trees keep 1 slot.
                        promoted = getattr(self, "_npu_rtree_promoted", {}) or {}
                        slots = promoted.get(tree.prefix)
                        ndim += len(slots) if slots else 1
            return ndim
        return sum(int(tree.tensor_dim is not None) for tree in self.range_trees)

    def dense_size_str(self):
        if triton_codegen_linearize:
            ndim = self.triton_tensor_ndim()
            if ndim == 0:
                return "[]"
            sizes = ["1"] * ndim
            for tree in self.range_trees:
                if tree.tensor_dim is None:
                    continue
                if not tree.is_reduction or self.inside_reduction:
                    sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"
            return f"[{', '.join(sizes)}]"
        sizes = self.dense_size_list()
        return f"[{', '.join(sizes)}]"

    def reduction_resize(self, value):
        if triton_codegen_linearize:
            if not self.no_x_dim and self.inside_reduction:
                ndims = self.triton_tensor_ndim()
                if ndims <= 1:
                    return f"triton_helpers.promote_to_tensor({value})"
                # Only deviate from the historical "None at last slot" layout
                # when the pre-body hook _npu_repermute_tensor_dims actually
                # moved the R-tree off the last slot. Touching this path for
                # un-permuted kernels has caused subtle slot mismatches in
                # multi-axis R-tree cases.
                if getattr(self, "_npu_tile_permuted", False):
                    r_slots = {
                        t.tensor_dim
                        for t in self.range_trees
                        if t.is_reduction and t.tensor_dim is not None
                    }
                    sizes = ["None" if i in r_slots else ":" for i in range(ndims)]
                else:
                    sizes = [":"] * (ndims - 1) + ["None"]
                return f"{value}[{', '.join(sizes)}]"
            return value
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})"
        sizes = [":"] * ndims
        sizes[-1] = "None"
        return f"{value}[{', '.join(sizes)}]"

    def reduction_resize_and_shape(self, value, shape):
        # 2.9.0 split reduction_resize() into reduction_resize_and_shape() (slices the
        # result AND reports post-reduction shape to CSE). sum/min/max/prod now route HERE
        # (only argmax/argmin hit reduction_resize), so our permuted override no longer
        # covers sum → upstream emits "[:, None]", broadcasting the OUTER store to [XBLOCK,
        # XBLOCK] (~100x). Mirror the permuted-slot logic, deviating ONLY in permuted-linearize.
        if (
            triton_codegen_linearize
            and not self.no_x_dim
            and self.inside_reduction
            and getattr(self, "_npu_tile_permuted", False)
        ):
            ndims = self.triton_tensor_ndim()
            if ndims > 1:
                r_slots = {
                    t.tensor_dim
                    for t in self.range_trees
                    if t.is_reduction and t.tensor_dim is not None
                }
                if r_slots:
                    sizes = ["None" if i in r_slots else ":" for i in range(ndims)]
                    new_value = f"{value}[{', '.join(sizes)}]"
                    new_shape = tuple(
                        1 if i in r_slots else (shape[i] if i < len(shape) else 1)
                        for i in range(ndims)
                    ) if shape is not None else None
                    return new_value, new_shape
        return super().reduction_resize_and_shape(value, shape)

    def _combine_contiguous_dims(self, index, tree):
        # In linearize mode each x-tree dim is a separate node (x0, x1, …) with its own
        # arange slice and mask. Upstream _combine_contiguous_dims() merges [x0(100),
        # x1(4)] into one flat x2(400), which has no per-node shape and breaks the
        # broadcast. Disable entirely in linearize mode.
        if triton_codegen_linearize:
            return index
        return super()._combine_contiguous_dims(index, tree)

    def indexing(self, index, *, copy_shape=None, dense_indexing=False,
                 override_mask=None, block_ptr=False,
                 tma_compatibility_checker=None, mask_constant_index=False):
        result = super().indexing(
            index, copy_shape=copy_shape, dense_indexing=dense_indexing,
            override_mask=override_mask, block_ptr=block_ptr,
            tma_compatibility_checker=tma_compatibility_checker,
            mask_constant_index=mask_constant_index,
        )
        if not triton_codegen_linearize:
            return result

        # Linearize mode emits per-node masks (x0mask, x1mask, …) and xmask = their AND,
        # but upstream indexing() always adds coarse "xmask" for any x-symbol. Rebuild
        # mask_vars from result.index.free_symbols: each x-tree symbol emits var.name+"mask"
        # ("x2mask") instead of "xmask"; r-tree symbols keep the prefix mask ("r0_mask").
        # Non-IndexingOptions (BlockPtrOptions) returned unchanged.
        if not isinstance(result, IndexingOptions):
            return result
        if override_mask:
            return result

        # Scalar (integer-index) load broadcast: upstream sets expand_str =
        # dense_size_str() = "[XBLOCK]", but the linearize tile is shaped by per-axis
        # real_block_xN constexprs, not XBLOCK. A (XBLOCK,) broadcast right-aligns against
        # a multi-axis tile → (rb_x3, rb_x2, XBLOCK) numel, exceeding Triton's 1M cap.
        # Use [1] — it broadcasts cleanly with any multi-axis tile.
        expand_str = result.expand_str
        if isinstance(index, sympy.Integer) and expand_str == self.dense_size_str():
            expand_str = "[1]"

        new_mask_vars = OrderedSet()
        # Pre-compute the inner reduction node per reduction tree. In nested-loop mode
        # only the innermost r-node is vectorized; outer r-nodes are Python scalars
        # (`for r0_2 in range(...)`). `{prefix}mask` covers the inner axis only, so an
        # index depending solely on an outer r-node must NOT carry it (else Triton can't
        # broadcast a [..., 1] address against a [..., R0_BLOCK] mask).
        inner_r_symbol_names = set()
        for tree in self.range_trees:
            if not tree.is_reduction:
                continue
            tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
            free_nodes = [
                n for n in tree.nodes.values()
                if n.name not in tree_node_mapping
            ]
            if len(free_nodes) < 2:
                # No split: all symbols share the single r-mask.
                for n in free_nodes:
                    inner_r_symbol_names.add(n.symbol().name)
                continue

            def _divisor_sort_key(n):
                if isinstance(n.divisor, (int, sympy.Integer)):
                    return int(n.divisor)
                return int(V.graph.sizevars.optimization_hint(n.divisor))

            inner = min(free_nodes, key=_divisor_sort_key)
            inner_r_symbol_names.add(inner.symbol().name)

        for var in result.index.free_symbols:
            assert isinstance(var, sympy.Symbol)
            if symbol_is_type(var, SymT.TMP):
                cse_var = self.cse.varname_map[var.name]
                new_mask_vars.update(cse_var.mask_vars)
            elif symbol_is_type(var, (SymT.UNBACKED_INT, SymT.SIZE,
                                      SymT.PRECOMPUTED_SIZE, SymT.INDEX,
                                      SymT.FLOAT, SymT.UNBACKED_FLOAT)):
                pass
            else:
                # x/y/z entry symbols (x0, x1, x2, …): use per-node mask
                # r-tree symbols: prefix-level mask (r0_mask, r1_mask, …) —
                # but only when the symbol corresponds to the innermost,
                # vectorized r-node. Outer r-nodes are scalar Python loop
                # vars and don't contribute to the mask.
                prefix_matches = [
                    prefix_str[symt]
                    for symt in TritonSymbols.block_types
                    if symbol_is_type(var, symt)
                ]
                if len(prefix_matches) == 1:
                    pfx = prefix_matches[0]
                    if pfx == 'r' or pfx.startswith('r'):
                        if var.name in inner_r_symbol_names:
                            new_mask_vars.add(f"{pfx}mask")
                    else:
                        new_mask_vars.add(f"{var.name}mask")

        if self._load_mask:
            new_mask_vars.add(self._load_mask)

        self.filter_masks(new_mask_vars)

        # When _load_mask references axes the pointer index doesn't (constant_pad_nd emits
        # ``tmp = x0 < c`` but a load indexes only x1), tl.load broadcasts the mask and any
        # mask-only axis fails. Forcing the index to dense_size_str() fixes that but turns a
        # contiguous load into a gathered broadcast_to the ascend backend lowers worse. Skip
        # when the mask's axis-masks ⊆ the index's own: load stays linear, OOB safety kept.
        new_index_str = result.index_str
        new_expand_str = expand_str
        if (
            self._load_mask is not None
            and not isinstance(index, sympy.Integer)
            and not isinstance(result, BlockPtrOptions)
        ):
            # Axis-masks (xNmask / rN_mask) the index itself contributes, derived
            # from result.index's own free symbols (computed into new_mask_vars
            # above, minus the load_mask we appended).
            index_axis_masks = OrderedSet(
                m for m in new_mask_vars
                if isinstance(m, str) and m != self._load_mask
            )
            # Axis-masks the load_mask predicate depends on.
            mask_axis_masks = OrderedSet(
                m for m in getattr(self._load_mask, "mask_vars", ())
                if isinstance(m, str)
            )
            mask_axes_covered = mask_axis_masks.issubset(index_axis_masks)
            # NPU_SKIP_REDUNDANT_BROADCAST=0 restores the upstream always-broadcast
            # behavior (for A/B: it always re-adds the dense broadcast_to even when
            # the mask axes are already covered by the index).
            skip_redundant = ncfg.skip_redundant_broadcast
            if not (mask_axes_covered and skip_redundant):
                dense = self.dense_size_str()
                if new_expand_str != dense:
                    new_index_str = f"tl.broadcast_to({new_index_str}, {dense})"
                    new_expand_str = dense

        if (
            new_mask_vars == result.mask_vars
            and new_expand_str == result.expand_str
            and new_index_str == result.index_str
        ):
            return result
        return IndexingOptions(
            new_index_str,
            new_mask_vars,
            new_expand_str,
            result._has_rindex,
            result.index,
            expand_shape=result.expand_shape,
        )

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        if not triton_codegen_linearize:
            return super().iteration_ranges_get_pid(entry)

        assert entry.grid_dim is not None
        # In linearize mode, (group_base + i) is the flat block index covering
        # all dimensions — yz grid overflow handling is not needed here.
        key = "(group_base + i)"
        pid = entry.pid_cache.get(key, key)
        if self.index_dtype != "tl.int32":
            return f"{pid}.to({self.index_dtype})"
        return pid

    def codegen_range_tree(self):
        """
        Replacement for TritonKernel.codegen_range_tree() in linearize mode.
        For non-r dimensions, calls our custom codegen_header_npu instead of the
        default iteration_ranges_codegen_header.
        """
        if not triton_codegen_linearize:
            return super().codegen_range_tree()

        # npu_header owns the large linearize header generator; import lazily to
        # avoid a load-time cycle (npu_header imports _FlatMapExpr from this module).
        from .npu_header import _codegen_header_npu_for_tree

        # In linearize mode all grid dimensions are flattened into (group_base+i).
        # For 2D+ pointwise kernels, higher grid_dim trees (e.g. y-tree) need to
        # first divide out the lower grid_dim tree's block count before taking %.
        # We accumulate outer_blocks across trees ordered by grid_dim so that
        # y0offset = (group_base+i) // x1_blocks % y0_blocks * real_block_y0.
        accumulated_blocks: list[str] = []
        for tree in self.range_trees:
            if not tree.is_loop:
                if not tree.is_reduction:
                    # Linearize mode: use NPU-specific header generator
                    outer = list(accumulated_blocks)
                    _codegen_header_npu_for_tree(self, tree, self.body, outer_blocks=outer)
                    # After processing this tree, append its node block names
                    # (in the same order as has_offset builds them) so that
                    # higher-dim trees will divide them out.
                    for node in tree.nodes.values():
                        if node.name not in tree.tree_node_mapping:
                            accumulated_blocks.append(f"{node.name}_blocks")
                else:
                    self.iteration_ranges_codegen_header(tree, self.body)
            elif self.inside_reduction:
                # r-tree loop workaround (gist.github.com/jansel/6527126f781559095c5531f98a4235a7):
                # bypass the NPU override of iteration_ranges_ranges_code (triton_tensor_ndim()
                # counts unmapped x-nodes, empty at __init__ (ndim=0) → crash). Build the r-tree
                # base arange directly: after _apply_linearize reassigns tensor_dim use
                # triton_tensor_ndim(), else fall back to orig_ndim.
                orig_ndim = sum(
                    int(t.tensor_dim is not None) for t in self.range_trees
                )
                npu_ndim = self.triton_tensor_ndim()
                total_ndim = max(orig_ndim, npu_ndim, tree.tensor_dim + 1)
                sizes = ["None"] * total_ndim
                sizes[tree.tensor_dim] = ":"
                index_dtype = self.index_dtype
                suffix = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
                ranges_code = f"tl.arange(0, {tree.prefix.upper()}BLOCK)[{', '.join(sizes)}]{suffix}"
                self.body.writeline(
                    f"{tree.prefix}base = {ranges_code}"
                )

        if self.inside_reduction:
            if any(tree.is_loop for tree in self.range_trees):
                rn_bases = self._get_reduction_symbols(
                    "base", integer=True, nonnegative=True
                )
                rbase = self._flatten_reduction_indices(rn_bases)
                self.body.splice(f"rbase = {self.index_to_str(rbase)}")
            else:
                self.codegen_reduction_indices(self.body)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        """
        Linearize mode: for non-r entries, skip default codegen entirely — their
        assignments (x0 = x0index) are emitted by _codegen_header_npu_for_tree.
        Only r-tree (loop) entries use the default body/indexing_code path.
        """
        if not triton_codegen_linearize:
            return super().codegen_iteration_ranges_entry(entry)

        if not entry.root.is_reduction:
            # x-tree entries: assignments emitted in codegen_range_tree header, skip here
            return
        # Default behavior for r-tree entries
        line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"
        if entry.root.is_loop:
            self.indexing_code.writeline(line)
        else:
            self.body.writeline(line)

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
        """
        Linearize mode: override ranges_code to use prefix.upper()BLOCK.

        In linearize mode, NPUTritonKernel.triton_tensor_ndim() counts only
        unmapped x-tree nodes plus r-trees — the upstream tensor_dim value
        (which allocates slot 0 for x, slot 1 for r0_) no longer matches the
        effective ndim.  Recompute the correct slot index here.

        After _apply_linearize has run (``_linearize_applied`` set), ``entry.tensor_dim``
        is already re-anchored to the effective slot — it is the same value
        dense_size_str()/tl.full()/the r-tree base arange use to lay out the tensor.
        Honor it directly. This matters for permuted reductions (e.g. sum(dim=0),
        where _npu_repermute_tensor_dims pushes the R-tree to slot 0 and the X
        sub-axes to outer slots): the declaration-order ``effective_dim`` walk below
        would always place the X slots first and wrongly put the reduction in the
        LAST slot, producing a header ``rbase`` whose broadcast shape is the
        transpose of the in-loop ``r0_base`` (the latter built from tensor_dim) —
        a loop-carried-type conflict when a free X-axis takes the inner-loop path.
        """
        if not triton_codegen_linearize:
            return super().iteration_ranges_ranges_code(entry)

        assert entry.tensor_dim is not None
        if getattr(self, "_linearize_applied", False):
            # tensor_dim is authoritative post-linearize; same slot dense_size_str uses.
            effective_dim = entry.tensor_dim
        else:
            # Pre-linearize: tensor_dim is still the stale upstream default, so
            # recompute the effective tensor slot from declaration order, matching
            # triton_tensor_ndim() (count the slots each preceding tree contributes).
            effective_dim = 0
            for tree in self.range_trees:
                if tree.tensor_dim is None:
                    continue
                if tree is entry:
                    break
                if tree.prefix != 'r':
                    # x/y/z tree: count unmapped nodes only
                    effective_dim += sum(
                        1 for node in tree.nodes.values()
                        if node.name not in getattr(tree, 'tree_node_mapping', {})
                    )
                else:
                    effective_dim += 1
        size = self.indexing_size_str(effective_dim)
        index_dtype = self.index_dtype
        convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
        return f"tl.arange(0, {entry.prefix.upper()}BLOCK){size}{convert}"

    def iteration_ranges_scalar_code(self, entry: IterationRangesRoot, value) -> str:
        """
        Linearize mode: override scalar_code.
        """
        if not triton_codegen_linearize:
            return super().iteration_ranges_scalar_code(entry, value)

        index_dtype = self.index_dtype
        ndim = self.triton_tensor_ndim()
        size = [1] * ndim
        if self.no_x_dim:
            return f"tl.full([1, 1], {value}, {index_dtype})"
        return f"tl.full({size}, {value}, {index_dtype})"

    def _npu_rsplit_rprefix(self) -> str:
        """Reduction-tree prefix (e.g. 'r0_') for the rsplit partial kernel."""
        for tree in self.range_trees:
            if tree.is_reduction:
                return tree.prefix
        return "r0_"

    def _npu_rsplit_x_total_numel(self) -> str:
        """Code string for the total x (output) element count: the product of
        all free non-reduction node lengths. Used as the per-core workspace row
        stride. Static lengths fold to ints; dynamic ones use <node>numel."""
        parts = []
        for tree in self.range_trees:
            if tree.is_reduction or tree.is_loop:
                continue
            tree_node_mapping = getattr(tree, "tree_node_mapping", {})
            for node in tree.nodes.values():
                if node.name in tree_node_mapping:
                    continue
                if isinstance(node.length, (int, sympy.Integer)):
                    parts.append(str(int(node.length)))
                else:
                    parts.append(f"{node.name}numel")
        if not parts:
            return "1"
        return " * ".join(parts)

    def _npu_rsplit_out_dtype(self):
        """Output dtype of the (single) reduction — the workspace element dtype.
        The partial sums are kept in this dtype (matches what the original store
        writes). Falls back to float32."""
        try:
            feats = self.features
            rnode = feats.reduction_nodes()[0]
            name = rnode.node.get_name()
            return V.graph.get_dtype(name)
        except Exception:
            return torch.float32

    def _npu_build_grid_recipe(self):
        """Build the host-side recipe the A5 launcher uses to reproduce the
        kernel's ``total_blocks`` (the exact program count for one-program-per-
        tile dispatch). Returns ``None`` when the recipe would not be usable
        (rsplit, no free x-axes, or a pre_loop line referencing a name the
        launcher cannot see), in which case dispatch falls back to group-based.

        The recipe is a dict::

            {"lines": [<python assignment strings>, ...],
             "factors": [<x>_blocks names in product order],
             "block_names": {<x>_blocks: True}}

        ``lines`` are the pre_loop block-count computations (real_block_<x>,
        <x>numel, <x>_blocks, the greedy x_g_tile chain, cumblk) with the
        ``: tl.constexpr`` annotation stripped so they exec as plain Python.
        They reference only ``XBLOCK`` (bound as a literal by the launcher) and
        the per-node ``<x>numel`` runtime args (present in the launcher's
        def_args). total_blocks = product(factors).
        """
        # rsplit uses total_thread multi-core cooperation, not one-per-tile.
        if self.npu_rsplit_partial:
            return None

        factors = []
        lines = []
        for tree in self.range_trees:
            if tree.is_reduction or tree.is_loop:
                continue
            pre = getattr(tree, 'pre_loop_code', None)
            if pre is not None and pre._lines:
                for entry in pre._lines:
                    line = entry if isinstance(entry, str) else getattr(entry, "line", None)
                    if not isinstance(line, str):
                        return None
                    # Strip the ``name : tl.constexpr = expr`` annotation to a
                    # plain ``name = expr`` so exec doesn't need tl in scope.
                    stripped = re.sub(r'\s*:\s*tl\.constexpr\s*=', ' =', line)
                    lines.append(stripped.strip())
            tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
            for node in tree.nodes.values():
                if node.name not in tree_node_mapping:
                    factors.append(f"{node.name}_blocks")

        if not factors:
            return None
        return {"lines": lines, "factors": factors}

    def _codegen_npu_dispatch_prologue(self, code):
        """Emit the per-kernel intra-core block->core dispatch prologue.

        Runs inside ``codegen_kernel``'s ``def <kernel>(...):`` body, before the
        ``for i in range(group_size):`` loop. In order it emits:
          1. the hoisted pre_loop_code (real_block_<x>, <x>_blocks) for each free
             x-tree, lifted out of the per-core loop;
          2. ``total_blocks`` = product of all free-node ``<x>_blocks`` (auxiliary
             flattened-view axes are already excluded upstream by tree_node_mapping);
          3. one of three dispatch shapes that set ``group_size`` / ``group_base``
             (the flat-tile range each core walks as ``(group_base + i)``).

        Only the normal branch's dispatch math lives here algorithmically; the
        rsplit and no-free-node shapes are emitted verbatim.
        """
        # 1. Hoist loop-invariant x-tree computations (real_block_x0, x0_blocks)
        # out of the "for i" loop to avoid redundant scalar div/mod per iteration.
        for tree in self.range_trees:
            if not tree.is_reduction and not tree.is_loop:
                pre = getattr(tree, 'pre_loop_code', None)
                if pre is not None and pre._lines:
                    code.splice(pre)

        # 2. total_blocks = product of all free-node _blocks. Auxiliary axes (a
        # node whose length is the product of other free nodes' lengths, i.e. a
        # flattened view) are excluded via tree_node_mapping so they don't
        # double-count.
        all_blocks_names = []
        for tree in self.range_trees:
            if not tree.is_reduction and not tree.is_loop:
                tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
                all_blocks_names.extend(
                    f"{n.name}_blocks" for n in tree.nodes.values()
                    if n.name not in tree_node_mapping
                )

        def emit_total_blocks():
            """Emit ``total_blocks = a * b * ...`` if needed; return the var/expr
            naming the product (a bare single name when there's only one factor,
            ``"1"`` when there are none)."""
            if not all_blocks_names:
                return "1"
            expr = " * ".join(all_blocks_names)
            if len(all_blocks_names) > 1:
                code.writeline(f"total_blocks = {expr}")
                return "total_blocks"
            return expr

        # 3. dispatch shape
        if self.npu_rsplit_partial:
            # r-axis cross-core split: program_id selects a contiguous chunk of the
            # reduction axis, not the x-axis. Every core walks ALL x-blocks (group_base=0,
            # group_size=total_blocks) so (group_base+i)==i enumerates every x tile, and
            # each core reduces only its own [r_lo, r_hi) slice (rewritten in the body).
            # Partials land in a per-core workspace row.
            total_blocks_var = emit_total_blocks()
            rprefix = self._npu_rsplit_rprefix()
            code.writeline("group_base = 0")
            code.writeline(f"group_size = {total_blocks_var}")
            code.writeline(
                f"{rprefix}_per_core = ({rprefix}numel + total_thread - 1) // total_thread"
            )
            code.writeline(f"{rprefix}_lo = group_id * {rprefix}_per_core")
            code.writeline(
                f"{rprefix}_hi = tl.minimum({rprefix}_lo + {rprefix}_per_core, {rprefix}numel)"
            )
        elif all_blocks_names and device_props.is_a5():
            # A5 (910_95): runtime schedules programs across cores, so launch one per tile,
            # skip the group loop. group_base==program_id, group_size==1. Odometer decodes
            # (group_base+i) modulo each <x>_blocks (period total_blocks), so the launcher
            # MUST size the grid to exactly total_blocks (npu_dispatch_recipe) or pid aliases;
            # over 65535 coreDim the backend folds logical→physical (all_blocks_parallel).
            code.writeline("group_size = 1")
            code.writeline("group_base = group_id")
        elif all_blocks_names:
            # Normal x-axis dispatch: spread total_blocks tiles over the 40 cores. The
            # first `group_tail` cores take one extra tile each, shifting every later
            # core's base by group_tail. Branch:
            #   tail core   -> group_size+1 tiles from group_id*(group_size+1)
            #   normal core -> group_size tiles from group_id*group_size + group_tail
            total_blocks_var = emit_total_blocks()
            code.writeline(f"group_size = {total_blocks_var} // total_thread")
            code.writeline(f"group_tail = {total_blocks_var} % total_thread")
            code.writeline("if group_id < group_tail:")
            code.writeline("    group_size = group_size + 1")
            code.writeline("    group_base = group_id * group_size")
            code.writeline("else:")
            code.writeline("    group_base = group_id * group_size + group_tail")
        else:
            # No free x-tree nodes (all mapped); each thread handles one iteration.
            code.writeline("group_size = 1")
            code.writeline("group_base = group_id")

    def codegen_kernel(self, name=None):
        """
        Override codegen_kernel to:
        1. Import npu_triton_heuristics instead of upstream triton_heuristics
        2. Add total_size arg for NPU 40CU group dispatch
        3. Wrap kernel body in group-based loop
        """
        from torch._inductor.utils import triton_version_uses_attrs_dict
        from torch._inductor.codegen.triton_utils import (
            config_of, signature_to_meta, equal_1_arg_indices, non_constexpr_signature
        )

        # r-axis cross-core split: allocate the per-core partial workspace BEFORE
        # python_argdefs() so ws_ptr shows up in the signature. It holds total_thread *
        # x_total partials (each core writes its own row), allocated with the OUTPUT dtype
        # so ws_ptr is already typed — NPU rejects the bitwidth-changing cast a uint8
        # workspace needs. zero_fill=False: every element is fully written, not accumulated.
        if self.npu_rsplit_partial and self.npu_rsplit_ws is None:
            from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode
            total_cores = device_props.get_npu_vector_core_count()
            out_dtype = self._npu_rsplit_out_dtype()
            x_total = self._npu_rsplit_x_total_numel()
            # count is in ELEMENTS (WorkspaceArg.count + dtype define the layout).
            x_numel_expr = sympy.Integer(1)
            for tree in self.range_trees:
                if tree.is_reduction or tree.is_loop:
                    continue
                x_numel_expr = x_numel_expr * tree.numel
            count = total_cores * x_numel_expr
            ws_arg = WorkspaceArg(
                count=count,
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                device=V.graph.get_current_device_or_throw(),
                outer_name=WorkspaceArg.unique_name(),
                inner_name="ws_ptr",
                dtype=out_dtype,
            )
            self.args.workspace_args.append(ws_arg)
            self.npu_rsplit_ws = (ws_arg.inner_name, 0, x_total, out_dtype)

        code = IndentedBuffer()

        size_hints = {}
        for prefix, numel in self.numels.items():
            if prefix_is_reduction(prefix) and not self.inside_reduction:
                continue
            # 2.13: sizevars.symbolic_hint was removed. The old call returned a
            # concrete int when statically/backed-resolvable and a symbolic expr
            # otherwise; guarding_hint_or_throw resolves backed symbols to their
            # hint int and raises only for unbacked (data-dependent) symbols, so
            # try/except-falling-back to 8192 preserves the original branch.
            try:
                size_hint = int(V.graph.sizevars.guarding_hint_or_throw(numel))
            except Exception:
                size_hint = 8192
            size_hints[prefix] = size_hint

        # Collect block hints for linearize mode
        block_hints = {}
        axis_hints = []
        if triton_codegen_linearize:
            for tree in self.range_trees:
                if tree.prefix != 'r':
                    block_hints[f"{tree.prefix.upper()}BLOCK_HINT"] = tree.get_block_hint()
                    tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
                    for node in tree.nodes.values():
                        if node.name in tree_node_mapping:
                            continue
                        length_hint = int(node.length) if isinstance(node.length, (int, sympy.Integer)) else -1
                        if isinstance(node.divisor, (int, sympy.Integer)):
                            divisor_hint = int(node.divisor)
                        else:
                            try:
                                divisor_hint = V.graph.sizevars.optimization_hint(node.divisor)
                            except Exception:
                                divisor_hint = -1
                            divisor_hint = int(divisor_hint) if isinstance(divisor_hint, int) and divisor_hint > 0 else -1
                        axis_hints.append({
                            "name": node.name,
                            "length": length_hint,
                            "divisor": divisor_hint,
                            "seed": 2 if divisor_hint > 1 and length_hint != 1 else 1,
                        })

        if name is None:
            code.splice(TritonKernel.gen_common_triton_imports())
            # Replace upstream triton_heuristics import with npu version
            code.splice(self.gen_triton_ext_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        # 2.7.1: python_argdefs returns (argdefs, call_args, signature, arg_types)
        argdefs, _, signature, _ = self.args.python_argdefs()

        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        mutated_args = OrderedSet()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                from torch._inductor.codegen.common import InplacedBuffer
                mutated_args.add(
                    cast(InplacedBuffer, self.args.inplace_buffers[mutation]).inner_name
                )
            if mutation in self.args.output_buffers:
                mutation_arg = self.args.output_buffers[mutation]
                from torch._inductor.codegen.common import RemovedArg
                assert not isinstance(mutation_arg, RemovedArg)
                mutated_args.add(mutation_arg)
        mutated_args = sorted(mutated_args)

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            argdefs.append(ArgName(sizearg.name))

            # Linearize mode: add per-node dynamic numel args. (Per-node divisor
            # is no longer emitted as a runtime arg: the greedy-via-unify tile
            # scheme computes real_block from XBLOCK + size hints and never
            # references <node>divisor in the body, so the arg was dead.)
            if tree.prefix != 'r' and triton_codegen_linearize:
                tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
                for node in tree.nodes.values():
                    if node.name in tree_node_mapping:
                        continue
                    if not isinstance(node.length, (int, sympy.Integer)):
                        node_sizearg = SizeArg(f"{node.name}numel", node.length)
                        signature.append(node_sizearg)
                        argdefs.append(ArgName(node_sizearg.name))

        # 2.7.1: signature_to_meta needs argdefs parameter
        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype, argdefs=argdefs
        )
        # NPU does not support fp64/i64; downcast to fp32/i32
        downcast_args = {}
        for k, v in triton_meta_signature.items():
            if v == "fp64":
                downcast_args[k] = v
                triton_meta_signature[k] = "fp32"
            elif v == "*fp64":
                downcast_args[k] = v
                triton_meta_signature[k] = "*fp32"
            elif v == "*i64":
                downcast_args[k] = v
                triton_meta_signature[k] = "*i32"

        triton_meta: dict = {
            "signature": triton_meta_signature,
            **({"downcast_args": downcast_args} if downcast_args else {}),
            # 2.7.1: DeviceProperties instead of raw device index
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
            "mix_mode": "aiv",  # NPU: force vector kernel generation
        }
        triton_meta["configs"] = [config_of(signature)]
        if triton_codegen_linearize:
            triton_meta['block_hints'] = block_hints
            triton_meta['axis_hints'] = axis_hints

        optimize_mem = V.graph.is_inference or V.graph.is_backward
        npu_num_x_nodes = 0
        if triton_codegen_linearize:
            for tree in self.range_trees:
                if not tree.is_reduction and not tree.is_loop:
                    tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
                    free_nodes = [n for n in tree.nodes.values()
                                  if n.name not in tree_node_mapping]
                    npu_num_x_nodes += len(free_nodes)
        inductor_meta = {
            "grid_type": self._get_grid_type().__name__,
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "optimize_mem": optimize_mem,
            "no_x_dim": self.no_x_dim,
            "num_load": self.num_load,
            "num_reduction": self.num_reduction,
            "npu_num_x_nodes": npu_num_x_nodes,
            **self.inductor_meta_common(),
        }
        # 2.13: upstream codegen_kernel stores inductor_meta on self so that
        # call_kernel (TritonKernel.call_kernel) can pass it to
        # generate_kernel_call. Mirror that here.
        self.inductor_meta = inductor_meta

        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        # 2.7.1: equal_1_arg_indices uses arg names as keys
        for arg_num in equal_1_arg_indices(signature):
            triton_meta["constants"][signature[arg_num].name] = 1

        # Mark static r-tree numel as constants so NPU compiler can prove
        # r0_mask = (r0_index < r0_numel) is always true when R0_BLOCK == r0_numel,
        # eliminating the scalar select/boundary-check path in the ttadapter.
        if triton_codegen_linearize:
            for tree in self.range_trees:
                if tree.is_reduction and isinstance(tree.numel, (int, sympy.Integer)):
                    numel_name = f"{tree.prefix}numel"
                    if any(getattr(s, 'name', None) == numel_name for s in signature):
                        triton_meta["constants"][numel_name] = int(tree.numel)

        self.triton_meta = triton_meta

        # Add BLOCK constexpr args
        def add_constexpr_arg(arg_name):
            if triton_version_uses_attrs_dict():
                signature.append(ConstexprArg(arg_name))
            argdefs.append(ArgName(arg_name, is_constexpr=True))

        for tree in self.range_trees:
            if tree.tensor_dim is None:
                continue
            add_constexpr_arg(f"{tree.prefix.upper()}BLOCK")

        self.codegen_body()

        # A5 one-program-per-tile: attach the host-side block-count recipe so the launcher
        # reproduces total_blocks (exact program count) and sizes the grid to it.
        # codegen_body() has populated pre_loop_code, so the recipe reads finished
        # block-count lines. A5 only (other chips keep group dispatch). Must run before the
        # heuristics decorator serializes inductor_meta.
        if triton_codegen_linearize and device_props.is_a5():
            _recipe = self._npu_build_grid_recipe()
            if _recipe is not None:
                inductor_meta["npu_dispatch_recipe"] = _recipe

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        # Build heuristics decorator — redirected to npu_triton_heuristics
        heuristics = self._get_heuristic()
        if self.inside_reduction:
            reduction_hint = self.features.get_reduction_hint()
            heuristics_line = f"""
                @npu_triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                tile_hint = "tile_hint=TileHint.SQUARE," if len(non_constexpr_signature(signature)) == 4 else "tile_hint=TileHint.DEFAULT,"  # noqa: B950
            heuristics_line = f"""
                @npu_triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(x.full_name() for x in argdefs)}):"
        )
        with code.indent():
            code.writeline(f"total_thread = {device_props.get_npu_vector_core_count()}")
            code.writeline("group_id = tl.program_id(0)")
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            if triton_codegen_linearize:
                # Emit the intra-core block->core dispatch prologue (pre_loop
                # hoists, total_blocks, group_size/group_base), then the body in
                # the per-core "for i" loop.
                self._codegen_npu_dispatch_prologue(code)
                code.writeline("for i in range(group_size):")
                with code.indent():
                    code.splice(self.body)
            else:
                code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        src = code.getvalue()
        if self.npu_rsplit_partial and self.npu_rsplit_ws is not None:
            src = self._npu_rewrite_rsplit_partial_body(src)
        # Strip degenerate broadcast_to that creates a block [1] from a scalar.
        # When xnumel=1, upstream emits "tmpN = tl.broadcast_to(tmpM, [1])" for
        # constant-index loads, but the store pointer is scalar — causing
        # "Value argument cannot be block type" errors on NPU triton.
        # Replace each occurrence by propagating the inner variable.
        marker = "tl.broadcast_to("
        suffix = ", [1])"
        while marker in src and suffix in src:
            idx = src.find(marker)
            end = src.find(suffix, idx)
            if end == -1:
                break
            inner = src[idx + len(marker):end]
            # Only strip if inner is a simple variable name (no nested parens)
            if inner.isidentifier():
                full_expr = marker + inner + suffix
                src = src.replace(full_expr, inner, 1)
            else:
                break
        # NPU has no warps — debug_barrier is unnecessary
        lines = src.split("\n")
        src = "\n".join(line for line in lines if "tl.debug_barrier()" not in line)

        return src

    def _npu_rewrite_rsplit_partial_body(self, src: str) -> str:
        """Rewrite the assembled partial-kernel source for the r-axis split,
        line by line with plain string ops (matching the linearize body-rewrite
        style; no regex):

          1. r-loop bound:  `for r0_offset in range(0, r0_numel, ...` -> `range(r0_lo, r0_hi, ...`
          2. r ragged mask: `... = r0_index < r0_numel`               -> `< r0_hi`
          3. output store:  `tl.store(out_ptr0 + (EXPR), V, M)`
                            -> `tl.store(ws_rs + (group_id * (x_total) + (EXPR)), V, M)`
             where `ws_rs = (ws_ptr + off).to(tl.pointer_type(<dt>))` is injected
             once just before the first redirected store.
        """
        ws_name, ws_offset, x_total, out_dtype = self.npu_rsplit_ws
        rprefix = self._npu_rsplit_rprefix()

        # The output buffer inner name (e.g. out_ptr0). Exactly one reduction
        # output for the rsplit path.
        from torch._inductor.codegen.common import RemovedArg
        out_ptr = None
        for buf_name, arg in self.args.output_buffers.items():
            if isinstance(arg, RemovedArg):
                continue
            inner = getattr(arg, "inner_name", arg)
            out_ptr = inner if isinstance(inner, str) else str(arg)
            break
        if out_ptr is None:
            log.debug("[NPU] rsplit partial: no output buffer found, skip rewrite")
            return src

        # Upstream emits the reduction loop start as ``... in tl.range(0, Nnumel,``
        # in 2.13 (older versions used the builtin ``range``). Match both forms;
        # the rewritten loop walks only this core's [lo, hi) slice with a plain
        # Python ``range`` over scalar bounds.
        loop_opens = (
            f"for {rprefix}offset in range(0, {rprefix}numel,",
            f"for {rprefix}offset in tl.range(0, {rprefix}numel,",
        )
        loop_open_new = f"for {rprefix}offset in range({rprefix}_lo, {rprefix}_hi,"
        mask_tok = f"{rprefix}index < {rprefix}numel"
        mask_tok_new = f"{rprefix}index < {rprefix}_hi"
        store_tok = f"tl.store({out_ptr} + ("

        # ws_ptr is already typed (allocated with out_dtype), so no pointer cast
        # is needed — NPU rejects bitwidth-changing pointer casts. Each core
        # writes its [x_total] partial into its own workspace row at offset
        # group_id * x_total.
        store_tok_new = f"tl.store({ws_name} + (group_id * ({x_total})) + ("

        new_lines = []
        for line in src.split("\n"):
            stripped = line.lstrip()
            # 1) r-loop bound
            for loop_open in loop_opens:
                if loop_open in line:
                    line = line.replace(loop_open, loop_open_new, 1)
                    break
            # 2) ragged-tail mask
            if mask_tok in line:
                line = line.replace(mask_tok, mask_tok_new)
            # 3) output store -> workspace
            if store_tok in stripped:
                line = line.replace(store_tok, store_tok_new, 1)
            new_lines.append(line)
        src = "\n".join(new_lines)

        log.debug("[NPU] rsplit partial rewrite: out_ptr=%s ws=%s off=%s x_total=%s rprefix=%s",
                  out_ptr, ws_name, ws_offset, x_total, rprefix)
        return src


    def codegen_body(self):
        """
        Override codegen_body to generate nested reduction loops when the
        r-tree has been split into inner/outer nodes by _apply_linearize.
        This eliminates modulo/divide in address calculations.
        """
        r_info = getattr(self, 'linearize_info', None)
        if r_info is None:
            super().codegen_body()
            self._apply_npu_addr_text_subs()
            # Real-block multi-axis reduction rewrite: a promoted r-tree has no
            # linearize_info; super().codegen_body() just materialized the upstream flat
            # ``for r0_offset`` loop into self.body. Rewrite here into per-node real_block
            # tile aranges + reshape-collapsed tl.sum. The loop is absent at _apply_linearize
            # time, so this later hook is correct. Guarded to fire once.
            rw = getattr(self, '_npu_rtree_rewrite_info', None)
            if (
                rw is not None
                and getattr(self, "_npu_rtree_promoted", {})
                and not getattr(self, "_npu_rtree_rewrite_done", False)
            ):
                _npu_rewrite_promoted_rtree_body(
                    self, rw["real_sizes"], rw["real_ndim"],
                )
                self._npu_rtree_rewrite_done = True
            self._maybe_rewrite_select_lane_load()

            return

        # If buffers are empty, the body was already generated and lives in
        # self.body._lines (from _apply_linearize re-injection). Rewrite it
        # in-place to use nested loops.
        if not (
            self.indexing_code
            or self.loads
            or self.compute
            or self.stores
            or self.post_loop_combine
            or self.post_loop_store
        ):
            # Body already assembled — rewrite the flat r-loop in body._lines
            tree = r_info['tree']
            inner_node = r_info['inner_node']
            inner_len = r_info['inner_len']
            outer_nodes = r_info['outer_nodes']
            prefix = tree.prefix
            inner_name = inner_node.name

            # Check if nested loops already exist (avoid duplicate rewrite)
            has_outer_loop = any(
                isinstance(line, str) and any(
                    f"for {onode.name} in range(" in line for onode in outer_nodes
                )
                for line in self.body._lines
            ) if outer_nodes else False
            if has_outer_loop:
                return

            def _len_str(length, node):
                if isinstance(length, (int, sympy.Integer)):
                    return str(int(length))
                return f"{node.name}numel"

            inner_len_str = _len_str(inner_len, inner_node)
            extra_indent = "    " * len(outer_nodes)

            new_lines = []
            inside_r_loop = False
            for line in self.body._lines:
                if not isinstance(line, str):
                    if inside_r_loop:
                        from torch._inductor.codegen.common import DeferredLineBase
                        if isinstance(line, DeferredLineBase):
                            new_lines.append(line.with_prefix(extra_indent))
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                    continue
                stripped = line.strip()
                indent = line[:len(line) - len(line.lstrip())] if stripped else ""

                # Detect the flat r-loop start
                if not inside_r_loop and _is_flat_rloop_header(stripped, prefix):
                    cur_indent = indent
                    for onode in outer_nodes:
                        olen_str = _len_str(onode.length, onode)
                        new_lines.append(f"{cur_indent}for {onode.name} in range({olen_str}):")
                        cur_indent += "    "
                    new_lines.append(f"{cur_indent}for {prefix}offset in range(0, {inner_len_str}, {prefix.upper()}BLOCK):")
                    inside_r_loop = True
                    continue

                if inside_r_loop:
                    # Replace: r0_mask = r0_index < r0_numel
                    if f"{prefix}mask = {prefix}index < " in stripped:
                        new_lines.append(f"{indent}{extra_indent}{prefix}mask = {prefix}index < {inner_len_str}")
                        continue
                    # Replace: r0_1 = (r0_index % N) → r0_1 = r0_index
                    if f"{inner_name} = " in stripped and "%" in stripped:
                        new_lines.append(f"{indent}{extra_indent}{inner_name} = {prefix}index")
                        continue
                    # Rewrite a folded flat reduction node assignment
                    # (`r0_3 = r0_index`) into its sub-axis decomposition
                    # (`r0_3 = r0_1 + 32*r0_2`) so downstream loads that index
                    # by the flat node stay valid now that it is no longer an
                    # independent outer loop.
                    _flat_subs = getattr(self, "_npu_flat_rnode_subs", None)
                    if _flat_subs:
                        _fname = stripped.split(" = ", 1)[0] if " = " in stripped else None
                        if _fname in _flat_subs:
                            new_lines.append(f"{indent}{extra_indent}{_fname} = {_flat_subs[_fname]}")
                            continue
                    # Remove outer-node decomposition assignments: any
                    # `r0_2 = ...r0_index...` (with `//`, `%`, or simplified
                    # mixes) — the Python loop iterator already provides
                    # this scalar value, so the tensor decomposition would
                    # shadow it and break loop-carried-variable type checks.
                    skip = False
                    for onode in outer_nodes:
                        if (
                            f"{onode.name} = " in stripped
                            and f"{prefix}index" in stripped
                        ):
                            skip = True
                            break
                    if skip:
                        continue
                    # All other lines inside the loop get extra indent
                    new_lines.append(f"{extra_indent}{line}")
                    continue

                new_lines.append(line)

            self.body._lines = new_lines
            return

        tree = r_info['tree']
        inner_node = r_info['inner_node']
        inner_len = r_info['inner_len']
        outer_nodes = r_info['outer_nodes']
        prefix = tree.prefix

        # Render lengths as code strings (handles both int and sympy expressions).
        # For dynamic lengths, use the kernel parameter name ({node.name}numel).
        def _len_str(length, node):
            if isinstance(length, (int, sympy.Integer)):
                return str(int(length))
            return f"{node.name}numel"

        inner_len_str = _len_str(inner_len, inner_node)

        # Rewrite indexing_code: replace modulo/divide decomposition with
        # direct variable references from the nested loop structure.
        new_indexing = IndentedBuffer()
        inner_name = inner_node.name
        for line in self.indexing_code._lines:
            if not isinstance(line, str):
                new_indexing._lines.append(line)
                continue
            stripped = line.strip()
            skip = False
            # Replace r0_1 = (r0_index % 256) with r0_1 = r0_1_index
            if f"{inner_name} = " in stripped and "%" in stripped:
                new_indexing.writeline(f"{inner_name} = {inner_name}_index")
                skip = True
            # Rewrite a folded flat reduction node (`r0_3 = r0_index`) into its
            # sub-axis decomposition so loads indexing the flat node remain
            # valid now that it is not an independent outer loop.
            if not skip:
                _flat_subs = getattr(self, "_npu_flat_rnode_subs", None)
                if _flat_subs:
                    _fname = stripped.split(" = ", 1)[0] if " = " in stripped else None
                    if _fname in _flat_subs:
                        new_indexing.writeline(f"{_fname} = {_flat_subs[_fname]}")
                        skip = True
            # Remove outer-node decomposition (with `//`, `%`, or any
            # simplified form referencing r0_index) — defined by Python
            # loop iterator.
            if not skip:
                for onode in outer_nodes:
                    if (
                        f"{onode.name} = " in stripped
                        and f"{prefix}index" in stripped
                    ):
                        skip = True
                        break
            if not skip:
                new_indexing._lines.append(line)

        # Emit outer scalar loops for each outer r-node
        for level, onode in enumerate(outer_nodes):
            oname = onode.name
            olen_str = _len_str(onode.length, onode)
            with self.body.indent(offset=level):
                self.body.writeline(f"for {oname} in range({olen_str}):")

        outer_depth = len(outer_nodes)

        # Emit inner vectorized loop
        with self.body.indent(offset=outer_depth):
            self.body.writeline(
                f"for {prefix}offset in range(0, {inner_len_str}, {prefix.upper()}BLOCK):"
            )

        # Emit loop body inside all nested loops
        with self.body.indent(offset=outer_depth + 1):
            self.body.writeline(f"{prefix}index = {prefix}offset + {prefix}base")
            self.body.writeline(f"{prefix}mask = {prefix}index < {inner_len_str}")
            self.body.writeline(f"{inner_name}_index = {prefix}index")
            # A folded flat reduction node's decomp references the inner node by BARE
            # name (`r0_1 = r0_2 + 512*r0_3`). new_indexing preserves indexing_code order,
            # where the flat node's line precedes the inner's (r0_1 before r0_2) → r0_2
            # used before bound. Emit the bare binding HERE before new_indexing AND drop
            # the later duplicate, so r0_2 is defined exactly once before the expansion.
            _flat_subs_active = getattr(self, "_npu_flat_rnode_subs", None)
            if _flat_subs_active:
                self.body.writeline(f"{inner_name} = {inner_name}_index")
                _dedup = IndentedBuffer()
                for _l in new_indexing._lines:
                    if isinstance(_l, str) and _l.strip() == f"{inner_name} = {inner_name}_index":
                        continue
                    _dedup._lines.append(_l)
                new_indexing = _dedup
            self.body.splice(new_indexing)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)

        self.cse.invalidate(self.outside_loop_vars)
        tree.cache_clear()

        self.body.splice(self.post_loop_combine)
        self.body.splice(self.post_loop_store)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.post_loop_combine.clear()
        self.post_loop_store.clear()
        self._apply_npu_addr_text_subs()

    def _apply_npu_addr_text_subs(self):
        """Rewrite dual-decomp flat reconstructions inside load/store addresses.

        The secondary decomposition's flat reconstruction (e.g. `x2 + 16384*x3`)
        is algebraically equal to the basis flat index but, written as
        `(flat%m) + m*(flat//m)`, bishengir cannot recognize it as contiguous
        and degrades the access to a scalar gather/scatter. Substitute it with
        the basis flat text on tl.load/tl.store lines so the burst stays
        contiguous (i.e. linearization is preserved after the dual-decomp fold).
        Restricted to address lines — the dead `tmpN = x2 + ...` scalar-compute
        lines keep their original symbols. Idempotent.
        """
        addr_subs = getattr(self, "_npu_addr_text_subs", None)
        if not addr_subs:
            return
        from torch._inductor.utils import DeferredLineBase

        def _rewrite(text):
            if not (isinstance(text, str) and ("tl.load" in text or "tl.store" in text)):
                return text
            for recon_text, flat_text in addr_subs.items():
                if recon_text in text:
                    text = text.replace(recon_text, flat_text)
            return text

        new_lines = []
        for line in self.body._lines:
            # The permuted store into a reused buffer is emitted as a
            # DeferredLine (rendered later by codegen_kernel), not a plain str,
            # so rewrite its `.line` in place; plain-string loads are rebuilt.
            if isinstance(line, DeferredLineBase):
                line.line = _rewrite(line.line)
            elif isinstance(line, str):
                line = _rewrite(line)
            new_lines.append(line)
        self.body._lines = new_lines

    def _maybe_rewrite_select_lane_load(self):
        """Rewrite a stride-k select-lane load into a contiguous full load +
        register-level ``extract_slice``.

        A ``x[..., 0]`` select on the innermost *input* axis of size K collapses
        into a flat address ``K*x2 + ...`` with no coeff-1 leaf. bishengir sees a
        strided, sub-16B inner run and degrades the access to an element-granular
        gather (the freqs / row-broadcast slowness). Re-adding a local lane
        arange ``[0, K)`` in the innermost *tensor* slot makes the inner run
        contiguous (one StridedSlice DMA burst); ``extract_slice`` then squeezes
        the lane back out, restoring the original tile shape so every downstream
        op is byte-for-byte unchanged.

        Detection is done structurally at load() time on the sympy index (see
        ``_maybe_record_select_lane_load``), which records each qualifying load
        by its result-var name in ``self._npu_select_lane_loads``. This method
        only has to (a) find that exact line by ``lhs == var`` — no text
        classification, no affine re-parse — and (b) emit the rewrite, reusing
        the line's own already-rendered ptr/addr/mask so the address text is
        never reconstructed. Gated on NPU_SELECT_EXTRACT_SLICE, once per kernel.

        Only fires in real-block linearize mode (needs the per-slot tile tokens
        and ``var_tensor_dims``); non-linearize kernels are left untouched.
        """
        if not ncfg.select_extract_slice:
            return
        if getattr(self, "_npu_select_slice_done", False):
            return
        if self.inside_reduction:
            return
        if not (triton_codegen_linearize and getattr(self, "_linearize_applied", False)):
            return
        targets = getattr(self, "_npu_select_lane_loads", None)
        if not targets:
            return

        import re

        # slot -> node-name across all free (non-reduction) trees.
        slot_to_name = {}
        for tree in self.range_trees:
            if tree.is_reduction:
                continue
            for nm, slot in (getattr(tree, "var_tensor_dims", {}) or {}).items():
                slot_to_name[slot] = nm
        if not slot_to_name:
            return
        ndim = self.triton_tensor_ndim()
        inner_slot = ndim - 1
        inner_name = slot_to_name.get(inner_slot)
        if inner_name is None:
            return

        # Recover the actual per-node tile-size token from the emitted arange
        # lines (``x3index = x3offset + tl.arange(0, x3_blk)[...]``). The token
        # differs between static (``real_block_x3``) and dynamic-S (``x3_blk``)
        # axes, so it must be read back, not assumed.
        arange_re = re.compile(r"^\s*(\w+)index\s*=.*tl\.arange\(0,\s*([^\)]+)\)")
        name_to_token = {}
        for line in self.body._lines:
            if isinstance(line, str):
                am = arange_re.match(line)
                if am:
                    name_to_token[am.group(1)] = am.group(2).strip()

        # Build extract sizes once (real per-slot tile tokens; inner slot -> 1).
        sizes = []
        for s in range(ndim):
            if s == inner_slot:
                sizes.append("1")
                continue
            tok = name_to_token.get(slot_to_name.get(s))
            if tok is None:
                return  # a needed tile token is missing → leave loads untouched
            sizes.append(tok)
        offsets = ", ".join(["0"] * ndim)
        strides = ", ".join(["1"] * ndim)
        bcast = self.indexing_size_str(inner_slot)

        # Narrow parse of ONE known line: pull indent/ptr/addr/mask verbatim.
        # (lhs is already known to be a recorded target, so this only extracts
        # the pieces to reassemble — it does not decide whether to rewrite.)
        line_re = re.compile(
            r"^(\s*)(\w+)\s*=\s*tl\.load\((\w+)\s*\+\s*\(([^()]*)\)\s*,\s*(.*)\)\s*$"
        )

        new_lines = []
        counter = 0
        changed = False
        for line in self.body._lines:
            if not isinstance(line, str):
                new_lines.append(line)
                continue
            m = line_re.match(line)
            if not m or m.group(2) not in targets:
                new_lines.append(line)
                continue
            indent, lhs, ptr, addr, rest = m.groups()
            # Two cases, by whether the inner tile-slot node appears in the address:
            #   * Case B (strided SLICE, x[:, ::k]): inner axis kept, iterated at stride k,
            #     so IS in address (``k*x1+...``). Load k*-wide contiguous run, take every k-th.
            #   * Case A (select, x[..., 0]): inner axis absent/broadcast, squeeze to size 1.
            addr_leaves = set(re.findall(r"x\d+", addr))
            if inner_name in addr_leaves:
                # Case B (strided slice) reads K× the inner bytes, discards K-1/K — a net
                # loss for a bandwidth-bound plain slice (0.98× vs strided gather). Unlike
                # Case A, the dropped lanes are REAL distinct elements, no free burst to win.
                # Gated by select_extract_slice_strided (default ON): target is a strided
                # sub-16B run feeding a broadcast store (freqs); disable when slice BW wins.
                if not ncfg.select_extract_slice_strided:
                    new_lines.append(line)
                    continue
                rewritten = self._emit_strided_slice_extract(
                    indent, lhs, ptr, addr, rest, inner_name, inner_slot,
                    ndim, slot_to_name, name_to_token, bcast, offsets, counter,
                )
                if rewritten is None:
                    # Precondition unmet (not a clean k*inner term, missing tile
                    # token, or no inner mask to bound the widened tail) → leave
                    # the strided load untouched rather than risk truncation.
                    new_lines.append(line)
                else:
                    new_lines.extend(rewritten)
                    counter += 1
                    changed = True
                continue
            lane = targets[lhs]

            lane_var = f"_es_lane{counter}"
            full_var = f"_es_full{counter}"
            counter += 1
            new_lines.append(f"{indent}{lane_var} = tl.arange(0, {lane}){bcast}")
            new_lines.append(
                f"{indent}{full_var} = tl.load({ptr} + ({addr} + {lane_var}), {rest})"
            )
            # Emit the slice via the registered first-class op (single source of
            # truth for the extract_slice text; see NPUTritonKernelOverrides).
            slice_expr = self.overrides.extract_slice(
                full_var,
                "[" + offsets + "]",
                "[" + ", ".join(sizes) + "]",
                "[" + strides + "]",
            )
            new_lines.append(f"{indent}{lhs} = {slice_expr}")
            changed = True

        if changed:
            self.body._lines = new_lines
        self._npu_select_slice_done = True

    def _emit_strided_slice_extract(
        self, indent, lhs, ptr, addr, rest, inner_name, inner_slot,
        ndim, slot_to_name, name_to_token, bcast, offsets, counter,
    ):
        """Case B: rewrite a strided SLICE load (``x[:, ::k]`` — inner axis kept
        and iterated at stride k) into a k*-wide CONTIGUOUS load + a stride-k
        ``extract_slice``.

        The strided address is ``k*<inner> + <rest-of-addr>``; bishengir sees a
        sub-16B strided inner run and degrades to an element gather (same root
        cause as the select case). Instead we:

          * widen the inner arange to ``k * real_block_inner`` contiguous lanes,
          * drop the ``k*<inner>`` term from the base and re-add ``k*<inner>offset``
            plus the widened lane so the DMA is one contiguous burst,
          * bound the widened tail with a fresh mask ``k*off + lane < k*numel``
            (the ``extract_slice`` then keeps every k-th lane), and
          * ``extract_slice`` with the inner stride set to k and the inner size
            kept at its real tile block, so the output axis is unchanged.

        Returns the replacement line list, or ``None`` if a precondition is
        unmet (no clean ``k*inner`` term, missing tile token, or no inner mask
        to bound the widened tail) — the caller then leaves the load as-is.
        """
        import re

        inner_tok = name_to_token.get(inner_name)
        if inner_tok is None:
            return None

        # Split the flat address into ``+``-separated terms and pull out the one
        # carrying the inner node, recovering its integer coefficient k.
        terms = [t.strip() for t in addr.split("+")]
        inner_term_idx = None
        k = None
        tok_re = re.compile(rf"^(?:(\d+)\s*\*\s*)?{re.escape(inner_name)}$")
        for i, t in enumerate(terms):
            tm = tok_re.match(t)
            if tm:
                inner_term_idx = i
                k = int(tm.group(1)) if tm.group(1) else 1
                break
        if inner_term_idx is None or k is None or k not in (2, 4, 8):
            return None  # inner node not a clean scalar-multiple term, or unit

        # Base address = all terms except the inner one, plus k*<inner>offset so
        # the contiguous burst still starts at this block's inner base.
        base_terms = [t for i, t in enumerate(terms) if i != inner_term_idx]
        base_addr = " + ".join(base_terms) if base_terms else "0"

        # The inner mask token (e.g. ``x1mask``) must be present in ``rest`` so we
        # can replace it with a widened-tail mask. Without it we cannot safely
        # bound the extra lanes the k*-wide load pulls in.
        inner_mask_tok = f"{inner_name}mask"
        if inner_mask_tok not in rest:
            return None

        # extract_slice geometry: inner slot keeps its real tile block but with
        # stride k; every other slot is identity (size = its tile token, stride 1).
        sizes = []
        strides = []
        for s in range(ndim):
            if s == inner_slot:
                sizes.append(inner_tok)
                strides.append(str(k))
                continue
            tok = name_to_token.get(slot_to_name.get(s))
            if tok is None:
                return None
            sizes.append(tok)
            strides.append("1")

        lane_var = f"_es_slane{counter}"
        full_var = f"_es_sfull{counter}"
        fmask_var = f"_es_smask{counter}"

        # Widened contiguous lane arange over k * real_block_inner elements.
        # Bound the tail: absolute inner position k*off + lane < k*numel keeps
        # every kept (k-th) lane in-row and masks the dropped tail lanes so the
        # widened load never reads past the row on the last block.
        inner_off = f"{inner_name}offset"
        inner_numel = f"{inner_name}numel"
        widened_mask = f"({k}*{inner_off} + {lane_var}) < {k}*{inner_numel}"
        # Substitute the inner mask token in ``rest`` with the widened mask
        # (word-boundary so x1mask does not clobber x12mask).
        rest_sub = re.sub(rf"\b{re.escape(inner_mask_tok)}\b", fmask_var, rest)

        out = [
            f"{indent}{lane_var} = tl.arange(0, {k}*{inner_tok}){bcast}",
            f"{indent}{fmask_var} = {widened_mask}",
            f"{indent}{full_var} = tl.load({ptr} + "
            f"({base_addr} + {k}*{inner_off} + {lane_var}), {rest_sub})",
        ]
        slice_expr = self.overrides.extract_slice(
            full_var,
            "[" + offsets + "]",
            "[" + ", ".join(sizes) + "]",
            "[" + ", ".join(strides) + "]",
        )
        out.append(f"{indent}{lhs} = {slice_expr}")
        return out



    def add_numel_to_call_args(self, name, call_args, arg_types):
        """
        Override to add per-tree numel args, plus per-node dynamic
        numel/divisor args in linearize mode.
        In 2.7.1 the method is add_numel_to_call_args (no _and_grid suffix).
        """
        if not triton_codegen_linearize:
            # Non-linearize: upstream behavior
            for tree in self.range_trees:
                expr = tree.numel if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)) else V.graph.wrapper_code.generate_numel_expr(name, tree)  # noqa: B950
                if not tree.is_reduction or self.inside_reduction:
                    call_args.append(expr)
                    arg_types.append(type(expr))
            return

        # Linearize mode: per-tree numel, then per-node dynamic numel/divisor
        for tree in self.range_trees:
            expr = tree.numel if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)) else V.graph.wrapper_code.generate_numel_expr(name, tree)  # noqa: B950

            if not tree.is_reduction or self.inside_reduction:
                call_args.append(expr)
                arg_types.append(type(expr))

            # Per-node dynamic numel for x trees. (Per-node divisor is no longer
            # passed: the greedy-via-unify tile scheme never references
            # <node>divisor in the kernel body, so it was a dead runtime arg.)
            if tree.prefix != 'r':
                tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
                for node in tree.nodes.values():
                    if node.name in tree_node_mapping:
                        continue
                    if not isinstance(node.length, (int, sympy.Integer)):
                        wrapper = V.graph.wrapper_code
                        var_name = f"{name}_{node.name}numel"
                        if (var_name, V.graph) not in wrapper.kernel_numel_expr:
                            wrapper.kernel_numel_expr.add((var_name, V.graph))
                            wrapper.writeline(
                                f"{wrapper.declare}{var_name} = "
                                f"{wrapper.codegen_python_sizevar(node.length)}{wrapper.ending}"
                            )
                        else:
                            wrapper.writeline(
                                f"{var_name} = "
                                f"{wrapper.codegen_python_sizevar(node.length)}{wrapper.ending}"
                            )
                        call_args.append(SymbolicCallArg(var_name, node.length))
                        arg_types.append(type(node.length))


def _npu_rsplit_outer_applicable(kernel) -> bool:
    """True iff the partial+combine r-axis cross-core split should fire for this
    kernel. Conditions (all required):
      1. We're inside an OUTER reduction (reduction_hint == OUTER).
      2. Exactly one reduction in the schedule, not a welford (multi-output sum
         is fine; welford has tuple semantics that the simple workspace+combine
         path doesn't model).
      3. The free (non-reduction) x-axis size hint product is < total cores —
         i.e. the upstream x-axis core split CAN'T fill all 40 cores.
      4. The reduction numel hint is large enough that paying for the second
         kernel launch is worthwhile.
      5. NPU_RSPLIT_OUTER env gate is on (default).
    """
    if not npu_rsplit_outer:
        return False
    if not getattr(kernel, "inside_reduction", False):
        return False
    feats = getattr(kernel, "features", None)
    if feats is None:
        return False
    try:
        if feats.get_reduction_hint() != ReductionHint.OUTER:
            return False
    except Exception:
        return False
    rnodes = feats.reduction_nodes()
    if len(rnodes) != 1:
        return False
    rnode = rnodes[0]
    try:
        if is_welford_reduction(rnode.node.data.reduction_type):
            return False
    except Exception:
        return False

    total_cores = device_props.get_npu_vector_core_count()
    # x (output) size hint: product of all non-reduction free node lengths.
    x_hint_product = 1
    for tree in kernel.range_trees:
        if tree.is_reduction or tree.is_loop:
            continue
        tree_node_mapping = getattr(tree, "tree_node_mapping", {})
        for node in tree.nodes.values():
            if node.name in tree_node_mapping:
                continue
            try:
                length_hint = V.graph.sizevars.optimization_hint(node.length)
            except Exception:
                length_hint = -1
            if not isinstance(length_hint, int) or length_hint <= 0:
                # Can't reason about it — bail out conservatively.
                return False
            x_hint_product *= length_hint

    try:
        rnumel_hint = V.graph.sizevars.optimization_hint(feats.reduction_numel)
    except Exception:
        return False
    if not isinstance(rnumel_hint, int) or rnumel_hint <= 0:
        return False

    # Trigger on the slow OUTER pattern: small output, dominant reduction axis. Small x
    # collapses the x-split to few blocks (autotune favors large XBLOCK), idling most
    # cores while each busy core serially walks the long strided r-axis. r-axis split
    # fills all cores and makes each load a contiguous x-wide burst. Conditions: x small
    # enough to under-fill cores, r >= 2048 (worth a 2nd launch), r dominates (r >= x).
    if x_hint_product > total_cores * 256:
        return False
    if rnumel_hint < 2048:
        return False
    if rnumel_hint < x_hint_product:
        return False

    return True


class NPUTritonScheduling(TritonScheduling):
    kernel_type = NPUTritonKernel

    def __init__(self, scheduler):
        super().__init__(scheduler)

    def create_kernel_choices(self, kernel_features, kernel_args, kernel_kwargs):
        """Override to always use NPUTritonKernel."""
        return [self.kernel_type(*kernel_args, **kernel_kwargs)]

    def _npu_writes_permuted_output(self, node):
        """True iff `node` writes an output buffer with a non-contiguous,
        permuted stride — the layout signature of a transpose/permute store.

        This is the fuse-time characteristic of the T5 position-bias backward
        scatter kernel: a softmax-bw + sum(0) reduction (producer, writes a
        CONTIGUOUS [H,Sq,Sk] buffer) is horizontally/vertically fused with the
        permute+where+index_put consumer, which writes the SAME data through a
        permuted view [Sq,Sk,H] stride [S,1,S**2]. Once fused into one kernel
        the permuted write becomes a div/mod scatter store (every cache line
        only 1/H utilized) and dominates runtime ~8x over the equivalent
        unfused path (contiguous reduction + a cheap permuted copy).

        Detect it structurally from the output Layout: a stride vector that is
        NOT a descending contiguous chain (i.e. some inner dim does not have
        the smallest stride) means the store is permuted. Available at fuse
        time (no kernel body needed). Symbolic strides are compared via the
        sizevar optimization hints so dynamic shapes are handled.
        """
        sizevars = V.graph.sizevars
        scheduler = self.scheduler
        name_to_buf = getattr(scheduler, "name_to_buf", {})
        for sub in node.get_nodes():
            rw = getattr(sub, "read_writes", None)
            if rw is None:
                continue
            for dep in rw.writes:
                buf = name_to_buf.get(dep.name)
                ir_node = getattr(buf, "node", None) if buf is not None else None
                if ir_node is None:
                    continue
                try:
                    layout = ir_node.get_layout()
                    strides = list(layout.stride)
                    sizes = list(layout.size)
                except Exception:
                    continue
                if len(strides) < 2:
                    continue
                # Hint-evaluate the symbolic strides; collapse size-1 dims
                # (their stride is arbitrary and must not flag a permute).
                pairs = [
                    (int(sizevars.optimization_hint(st)), int(sizevars.optimization_hint(sz)))
                    for st, sz in zip(strides, sizes)
                ]
                eff = [st for st, sz in pairs if sz != 1]
                if len(eff) < 2:
                    continue
                # Contiguous (row-major) layout has strictly descending strides
                # with the innermost == 1. A permuted store breaks that order.
                if eff[-1] != 1 or any(eff[i] < eff[i + 1] for i in range(len(eff) - 1)):
                    return True
        return False

    def can_fuse(self, node1, node2):
        """Gate fusion on the *combined read (load) count*, not just node count.

        The upstream caps (max_fusion_size, realize_opcount_threshold,
        realize_acc_reads_threshold) all measure either the number of scheduler
        nodes or the read count of a *single* node. None of them bound the
        loads of the FUSED kernel. The T5 position-bias backward exposes this:
        many independent softmax-backward subgraphs share the same attention
        operands and all reduce into one [H,Sq,Sk] buffer, so the scheduler
        horizontally fuses them. Each subgraph is a tiny node (2-3 read deps),
        so every per-node cap passes — yet 8 fused nodes produce a kernel with
        ~20 input pointers and ~86 loads, whose ~700-line body makes bishengir
        compile time explode.

        We add a backend gate: once the union of read dependencies across the
        would-be-fused node set exceeds NPU_MAX_FUSED_READS, refuse the fuse.
        This bounds the per-kernel load count directly (read deps are
        monotonic with emitted loads), splitting the monster into several
        small kernels each fast to compile. Gated by NPU_MAX_FUSED_READS
        (default 24; set 0 / unset-high to disable). Fusion is a pure perf
        heuristic, so refusing here never affects correctness.
        """
        if not super().can_fuse(node1, node2):
            return False

        # Don't constrain templates (matmul/attention prologue-epilogue fusion)
        # — those are handled by their own heuristics and aren't the blowup.
        for n in (node1, node2):
            if any(sub.is_template() for sub in n.get_nodes()):
                return True

        # Permuted-store guard (no_fuse_permuted_store, default on). Refuse to fuse a
        # reduction producer with a consumer writing a permuted output: fusing collapses
        # the permuted copy into a div/mod scatter store that dominates runtime (T5
        # position-bias bwd ~8x). Kept separate, the reduction stays a contiguous burst
        # and the permute is a cheap copy. Pure perf heuristic — never affects correctness.
        if ncfg.no_fuse_permuted_store:
            has_reduction = any(
                sub.is_reduction()
                for n in (node1, node2)
                for sub in n.get_nodes()
            )
            if has_reduction and (
                self._npu_writes_permuted_output(node1)
                or self._npu_writes_permuted_output(node2)
            ):
                from torch._inductor.scheduler import WhyNoFuse
                WhyNoFuse(node1, node2)(
                    "NPU: refusing reduction+permuted-store fuse (scatter store)"
                )
                return False

        max_reads = ncfg.max_fused_reads
        if max_reads <= 0:
            return True

        read_names = set()
        for n in (node1, node2):
            for sub in n.get_nodes():
                rw = getattr(sub, "read_writes", None)
                if rw is None:
                    continue
                for dep in rw.reads:
                    read_names.add(dep.name)

        if len(read_names) > max_reads:
            from torch._inductor.scheduler import WhyNoFuse
            WhyNoFuse(node1, node2)(
                "NPU: combined read count %d exceeds NPU_MAX_FUSED_READS=%d",
                len(read_names),
                max_reads,
            )
            return False
        return True

    # SIMDScheduling binds can_fuse_vertical/can_fuse_horizontal = can_fuse as CLASS-BODY
    # aliases pointing at SIMDScheduling.can_fuse, so overriding can_fuse here does NOT
    # redirect them. The Scheduler dispatches through those two, so without re-binding our
    # gate is never consulted. Re-alias both to this subclass's can_fuse.
    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures):
        """
        2.7.1: codegen_node_schedule receives SIMDKernelFeatures instead of
        (node_schedule, buf_accesses, numel, reduction_numel).
        """
        node_schedule = kernel_features.node_schedule

        tiling = self.select_tiling(
            node_schedule, kernel_features.numel, kernel_features.reduction_numel
        )
        kernels = self.create_kernel_choices(
            kernel_features, [tiling], {"features": kernel_features}
        )
        # Pre-permute tensor_dim per range_tree from the actual memory strides of
        # node_schedule reads/writes. Must run BEFORE codegen_node_schedule_with_kernel:
        # that pass emits body lines (tl.full, tl.broadcast_to, tl.sum) by stringifying
        # dense_size_str() at emit time, so a later tensor_dim change leaves stale shapes.
        # See _npu_repermute_tensor_dims.
        for kernel in kernels:
            self._npu_repermute_tensor_dims(kernel, kernel_features)
        for kernel in kernels:
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        if triton_codegen_linearize:
            self._apply_linearize(kernels[0] if len(kernels) == 1 else None, node_schedule)

        # r-axis cross-core split for OUTER reductions: when the x-axis core
        # split can't fill all 40 cores, flag the (single) kernel as the
        # "partial" stage. codegen_kernel then emits the program_id-as-r-chunk
        # form + per-core workspace store, and we generate a second "combine"
        # kernel below that sums the per-core partials into the real output.
        rsplit_kernel = None
        if len(kernels) == 1 and _npu_rsplit_outer_applicable(kernels[0]):
            kernels[0].npu_rsplit_partial = True
            rsplit_kernel = kernels[0]

        MultiKernel.merge_workspaces_inplace(kernels)
        for kernel in kernels:
            with V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, node_schedule, kernel)
            log.debug("Generating kernel code with kernel_name: %s", kernel_name)
            kernel.kernel_name = kernel_name
            from torch._inductor.codecache import code_hash
            kernel.code_hash = code_hash(src_code)
        del kernel

        # Build + register the combine kernel (literal template) once the
        # partial kernel has been emitted (so its workspace arg is finalized).
        combine_info = None
        if rsplit_kernel is not None and rsplit_kernel.npu_rsplit_ws is not None:
            combine_info = self._npu_emit_rsplit_combine(rsplit_kernel, node_schedule)
            if combine_info is None:
                # Couldn't build combine — fall back to single-kernel path by
                # clearing the flag would require re-codegen; instead error loud
                # in debug, else just proceed (partial wrote workspace, but no
                # combine — that would be wrong, so disable rsplit entirely).
                rsplit_kernel = None

        final_kernel: Any
        if len(kernels) > 1:
            final_kernel = MultiKernel(kernels)
        else:
            (final_kernel,) = kernels

        with V.set_kernel_handler(final_kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()

        self.codegen_comment(node_schedule)
        if rsplit_kernel is not None and combine_info is not None:
            self._npu_call_rsplit_kernels(rsplit_kernel, combine_info)
        else:
            final_kernel.call_kernel(final_kernel.kernel_name)

        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernels[0].kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
            V.graph.wrapper_code.supports_intermediate_hooks
            and config.generate_intermediate_hooks
        ):
            live_outs = kernels[0].args.live_output_buffers()
            for node in kernel_features.scheduler_nodes():
                name = node.get_name()
                if name not in live_outs:
                    continue
                assert node.node is not None
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.free_buffers_in_scheduler()

    def _npu_emit_rsplit_combine(self, partial, node_schedule):
        """Build + register the combine kernel for the r-axis split.

        The combine reads the per-core partials workspace[total_thread, x_total]
        and sums over the total_thread axis into the real output[x_total]. This
        is a tiny static OUTER reduction (reduce dim == total_thread == 40), so
        its loads are fully contiguous and it never hits the dynamic-mask
        non-linearization. Emitted as a literal templated kernel (the shape is
        fixed) and registered via define_kernel.

        Returns a dict with the info needed to launch it, or None on failure.
        """
        ws = getattr(partial, "npu_rsplit_ws", None)
        if ws is None:
            return None
        ws_name, ws_offset, x_total, out_dtype = ws
        total_cores = device_props.get_npu_vector_core_count()

        # Real output buffer (outer graph name) + its inner arg name.
        from torch._inductor.codegen.common import RemovedArg
        out_outer = None
        for outer, inner in partial.args.output_buffers.items():
            if isinstance(inner, RemovedArg):
                continue
            out_outer = outer
            break
        if out_outer is None:
            return None

        # x_total as an int when static (the common case); else symbolic name.
        x_numel_expr = sympy.Integer(1)
        for tree in partial.range_trees:
            if tree.is_reduction or tree.is_loop:
                continue
            x_numel_expr = x_numel_expr * tree.numel
        x_total_hint = V.graph.sizevars.optimization_hint(x_numel_expr)

        combine_name = str(Placeholder.KERNEL_NAME)
        desc_name = str(Placeholder.DESCRIPTIVE_NAME)
        # x size hint for the heuristic; r is the static core count.
        size_hints = {"x": int(x_total_hint), "r0_": total_cores}
        from torch._inductor.codegen.triton_utils import _type_of
        dt_star = _type_of(out_dtype)  # e.g. "*fp32"
        # Build the AttrsDescriptor config directly (config_of() resolves alignment via
        # scheduler.name_to_buf, but our "in_ptr0"/"out_ptr0" aren't real graph buffers).
        # Mark pointers (0,1) and static xnumel (2) 16-divisible: workspace+output are
        # fresh aligned allocations and xnumel is a multiple of 16. r0_numel (3) is the
        # constexpr core count.
        from triton.compiler.compiler import AttrsDescriptor
        div16 = [0, 1]
        if int(x_total_hint) % 16 == 0:
            div16.append(2)
        combine_attrs = AttrsDescriptor.from_dict({
            "arg_properties": {"tt.divisibility": tuple(div16), "tt.equal_to": ()},
            "cls": "AttrsDescriptor",
        })
        triton_meta = {
            "signature": {
                "in_ptr0": dt_star,
                "out_ptr0": dt_star,
                "xnumel": "i32",
                "r0_numel": "i32",
            },
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {"r0_numel": total_cores},
            "mix_mode": "aiv",
            "configs": [combine_attrs],
        }
        inductor_meta = {
            "grid_type": "Grid1D",
            "autotune_hints": set(),
            "kernel_name": desc_name,
            "mutated_arg_names": [],
            "optimize_mem": V.graph.is_inference or V.graph.is_backward,
            "no_x_dim": False,
            "num_load": 1,
            "num_reduction": 1,
            "npu_num_x_nodes": 1,
            **partial.inductor_meta_common(),
        }

        # Literal combine source. x is a single linear axis split across cores
        # (standard NPU group dispatch on x); the reduction axis is the static
        # core count. Mirrors the validated hand-written combine.
        imports = TritonKernel.gen_common_triton_imports() + partial.gen_triton_ext_imports()
        src = f'''
{imports}
@npu_triton_heuristics.reduction(
    size_hints={size_hints!r},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={triton_meta!r},
    inductor_meta={inductor_meta!r}
)
@triton.jit
def {combine_name}(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    total_thread = {total_cores}
    group_id = tl.program_id(0)
    xnumel = {int(x_total_hint)}
    x0numel : tl.constexpr = {int(x_total_hint)}
    real_block_x0 : tl.constexpr = x0numel if x0numel <= XBLOCK else XBLOCK
    x0_blocks : tl.constexpr = (x0numel + real_block_x0 - 1) // real_block_x0
    group_size = x0_blocks // total_thread
    group_tail = x0_blocks % total_thread
    group_base = group_id * group_size + group_tail
    if group_id < group_tail:
        group_size = group_size + 1
    if group_id < group_tail:
        group_base = group_id * group_size
    for i in range(group_size):
        x0offset = (group_base + i) * real_block_x0
        x0index = x0offset + tl.arange(0, real_block_x0)[None, :]
        x0 = x0index
        x0mask = x0index < x0numel
        _acc = tl.full([R0_BLOCK, real_block_x0], 0, tl.float32)
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + tl.arange(0, R0_BLOCK)[:, None]
            r0_mask = r0_index < r0_numel
            tmp0 = tl.load(in_ptr0 + (x0 + {int(x_total_hint)}*r0_index), r0_mask & x0mask, eviction_policy='evict_first', other=0.0).to(tl.float32)  # noqa: B950
            tmp1 = _acc + tmp0
            _acc = tl.where(r0_mask & x0mask, tmp1, _acc)
        tmp2 = tl.sum(_acc, 0)[None, :]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
'''
        combine_kernel_name = self.define_kernel(src, node_schedule, partial)
        return {
            "kernel_name": combine_kernel_name,
            "ws_name": ws_name,
            "ws_offset": ws_offset,
            "out_outer": out_outer,
            "x_total_hint": int(x_total_hint),
            "total_cores": total_cores,
            "out_dtype": out_dtype,
            "triton_meta": triton_meta,
        }

    def _npu_call_rsplit_kernels(self, partial, combine_info):
        """Emit the two-stage launch into the wrapper:
            alloc workspace
            partial(... , ws)        # writes per-core partials
            combine(ws, real_out)    # sums partials -> real output
            free workspace
        We bypass each kernel's auto alloc/dealloc (which would free the
        workspace between the two launches) and manage it once around both.
        """
        wrapper = V.graph.wrapper_code
        wrapper.write_triton_header_once()

        # Partial launch args (includes the workspace outer_name + real output,
        # the latter unused by the partial but harmless).
        _, p_call_args, _, p_arg_types = partial.args.python_argdefs()
        partial.add_numel_to_call_args(partial.kernel_name, p_call_args, p_arg_types)

        # Allocate workspace(s) once, before the partial.
        for wsarg in partial.args.workspace_args:
            wrapper.generate_workspace_allocation(wsarg)

        wrapper.generate_kernel_call(
            partial.kernel_name,
            p_call_args,
            triton=True,
            arg_types=p_arg_types,
            triton_meta=partial.triton_meta,
        )

        # Combine launch: explicit args [ws_ptr, real_out, xnumel, r0_numel].
        ws_outer = None
        for wsarg in partial.args.workspace_args:
            if wsarg.inner_name == combine_info["ws_name"]:
                ws_outer = wsarg.outer_name
                break
        if ws_outer is None and partial.args.workspace_args:
            ws_outer = partial.args.workspace_args[0].outer_name
        c_call_args = [
            ws_outer,
            combine_info["out_outer"],
            combine_info["x_total_hint"],
            combine_info["total_cores"],
        ]
        c_arg_types = [
            combine_info["out_dtype"],
            combine_info["out_dtype"],
            int,
            int,
        ]
        wrapper.generate_kernel_call(
            combine_info["kernel_name"],
            c_call_args,
            triton=True,
            arg_types=c_arg_types,
            triton_meta=combine_info["triton_meta"],
        )

        # Free workspace(s) once, after the combine.
        for wsarg in reversed(partial.args.workspace_args):
            wrapper.generate_workspace_deallocation(wsarg)

    def _expand_divmod_nodes(self, kernel, matcher):
        """Expand FloorDiv/Mod patterns on single-node trees into sub-nodes.

        When a single tree node y0 appears in body expressions as both
        (y0 // D) and (y0 % D), split it into two sub-nodes:
          inner (divisor=1, length=D)
          outer (divisor=D, length=numel/D)
        and replace the div/mod in the body text with the new variables.
        """
        from torch._inductor.utils import DeferredLineBase

        def _line_text(line):
            if isinstance(line, str):
                return line
            if isinstance(line, DeferredLineBase):
                return line.line
            return ""

        for tree in kernel.range_trees:
            if tree.is_reduction or tree.is_loop:
                continue
            free_nodes = [
                n for n in tree.nodes.values()
                if n.name not in tree.tree_node_mapping
            ]
            if len(free_nodes) < 1:
                continue

            # For pointwise kernels, body is empty at this stage.
            # Load/store code is in kernel.loads, kernel.compute, kernel.stores, kernel.suffix.
            all_lines = list(kernel.body._lines)
            for buf in (kernel.indexing_code, kernel.loads, kernel.compute, kernel.stores, kernel.suffix):
                if hasattr(buf, '_lines'):
                    all_lines.extend(buf._lines)
            body_text = "\n".join(_line_text(line) for line in all_lines)

            # Find a free node appearing as both FloorDiv(node, D) and
            # ModularIndexing(node, 1, D) — a merged axis re-split into a (D, length/D)
            # sub-grid (div/mod scatter load). Don't hand-write the version-dependent div/
            # mod spelling; RENDER each candidate via the texpr printer + substring-match.
            # Divisors come structurally from the length's factors (handles dynamic s17*s20).
            sizevars = V.graph.sizevars
            from torch.utils._sympy.functions import FloorDiv as SympyFloorDiv

            def _candidate_divisors(length):
                cands = []
                if isinstance(length, (int, sympy.Integer)):
                    L = int(length)
                    for D in range(2, L):
                        if L % D == 0:
                            cands.append(sympy.Integer(D))
                else:
                    # Symbolic: each proper sub-product of the Mul factors.
                    factors = list(sympy.Mul.make_args(length))
                    if len(factors) >= 2:
                        cands.extend(factors)
                return cands

            node = node_name = divisor_sym = None
            div_token = mod_token = mod_token2 = None
            outer_length = None
            for candidate in free_nodes:
                sym = sympy.Symbol(candidate.name)
                found = False
                for D in _candidate_divisors(candidate.length):
                    ol = SympyFloorDiv(candidate.length, D)
                    if not sizevars.statically_known_equals(D * ol, candidate.length):
                        continue
                    # Render the exact tokens the body would contain via the
                    # same printer (rename_indexing maps size symbols -> ks args).
                    dtok = texpr(kernel.rename_indexing(SympyFloorDiv(sym, D)))
                    mtok = texpr(kernel.rename_indexing(ModularIndexing(sym, sympy.Integer(1), D)))
                    if dtok in body_text and mtok in body_text:
                        node, node_name, divisor_sym = candidate, candidate.name, D
                        div_token, mod_token = dtok, mtok
                        mod_token2 = f"({mtok})"
                        outer_length = ol
                        found = True
                        break
                if found:
                    break
            if node is None:
                continue

            # Create two new sub-nodes relative to the original node's divisor
            # inner: same base divisor as node, length = divisor_sym
            # outer: divisor = node.divisor * divisor_sym, length = node.length / divisor_sym
            inner_entry = tree.lookup(node.divisor, divisor_sym)
            outer_entry = tree.lookup(node.divisor * divisor_sym, outer_length)

            if inner_entry.name == node_name or outer_entry.name == node_name:
                continue

            inner_name = inner_entry.name
            outer_name = outer_entry.name

            # Map original node
            node_expr = sympy.Symbol(inner_name) + divisor_sym * sympy.Symbol(outer_name)
            tree.tree_node_mapping[node_name] = node_expr
            pattern = f"{node_name} = {node_name}index"
            matcher[pattern] = f"{node_name} = {node_expr}"

            # Text replacement in all code buffers. `div_token` is the exact
            # floor-div substring matched above (either `(node // D)` or
            # `div_floor_integer(node,  D)`), so both forms are rewritten to the
            # new outer sub-node name.

            def _replace_in_buffer(buf):
                new_lines = []
                for line in buf._lines:
                    text = _line_text(line)
                    if not text:
                        new_lines.append(line)
                        continue
                    new_text = text.replace(div_token, outer_name)
                    new_text = new_text.replace(mod_token2, inner_name)
                    new_text = new_text.replace(mod_token, inner_name)
                    if new_text != text:
                        if isinstance(line, DeferredLineBase):
                            new_lines.append(line._new_line(new_text))
                        else:
                            new_lines.append(new_text)
                    else:
                        new_lines.append(line)
                buf._lines = new_lines

            _replace_in_buffer(kernel.body)
            for buf in (kernel.indexing_code, kernel.loads, kernel.compute, kernel.stores, kernel.suffix):
                if hasattr(buf, '_lines'):
                    _replace_in_buffer(buf)

            # Mark tree as expanded so _collapse_rowmajor_xtrees skips it
            tree._divmod_expanded = True

    def _collapse_rowmajor_xtrees(self, kernel, matcher):
        from torch._inductor.utils import DeferredLineBase

        def _line_text(line):
            if isinstance(line, str):
                return line
            if isinstance(line, DeferredLineBase):
                return line.line
            return ""

        for tree in kernel.range_trees:
            if tree.is_reduction or tree.is_loop:
                continue
            if getattr(tree, '_divmod_expanded', False):
                continue
            free_nodes = [
                n for n in tree.nodes.values()
                if n.name not in tree.tree_node_mapping
            ]
            if len(free_nodes) < 2:
                continue

            # Don't collapse if load addresses have a stride gap between x-nodes.
            # E.g. x0 + 768*r + 98304*x1: collapsing x0,x1 into x3 would require
            # mod/div to recover x0,x1 from x3 in the address computation.
            if kernel.inside_reduction:
                all_lines = list(kernel.body._lines)
                for buf in (kernel.indexing_code, kernel.loads, kernel.compute, kernel.stores):
                    if hasattr(buf, '_lines'):
                        all_lines.extend(buf._lines)
                body_text = "\n".join(_line_text(line) for line in all_lines)
                # Check if any reduction variable appears between x-node references
                # in a load/store address (indicates stride gap)
                has_stride_gap = False
                for n in free_nodes:
                    # If a node's stride in the address != product of inner lengths,
                    # there's a gap (reduction dim interleaved)
                    coeff_pattern = f"*{n.name}"
                    if coeff_pattern in body_text:
                        has_stride_gap = True
                        break
                if has_stride_gap:
                    continue

            # Path A: all lengths static — original collapse logic
            if all(
                isinstance(n.divisor, (int, sympy.Integer))
                and isinstance(n.length, (int, sympy.Integer))
                for n in free_nodes
            ):
                free_nodes.sort(key=lambda n: int(n.divisor))
                ok = int(free_nodes[0].divisor) == 1
                if ok:
                    expected = 1
                    for n in free_nodes:
                        if int(n.divisor) != expected:
                            ok = False
                            break
                        expected *= int(n.length)
                    if ok and not V.graph.sizevars.statically_known_equals(
                        sympy.Integer(expected), tree.numel
                    ):
                        ok = False
                if not ok:
                    continue
                flat_entry = tree.lookup(sympy.Integer(1), tree.numel)
                if flat_entry.name in {n.name for n in free_nodes}:
                    continue
                if flat_entry.name in tree.tree_node_mapping:
                    continue
                flat_sym = flat_entry.symbol()
                for n in free_nodes:
                    d = int(n.divisor)
                    L = int(n.length)
                    if d == 1 and L == int(tree.numel):
                        text = str(flat_sym)
                    elif d == 1:
                        text = f"({flat_sym} % {L})"
                    elif d * L == int(tree.numel):
                        text = f"({flat_sym} // {d})"
                    else:
                        text = f"(({flat_sym} // {d}) % {L})"
                    tree.tree_node_mapping[n.name] = _FlatMapExpr(
                        text, [flat_sym]
                    )
                    pattern = f"{n.name} = {n.name}index"
                    matcher[pattern] = f"{n.name} = {text}"
                continue

            # Path B: a flat node already exists among free_nodes (divisor=1,
            # length=numel) and the remaining nodes form a row-major
            # decomposition with static divisors. This handles the common
            # BERT pattern where a pos-embedding broadcast creates x0(N) +
            # x1(B) alongside the flat x3(B*N), all with dynamic B.
            if not all(isinstance(n.divisor, (int, sympy.Integer)) for n in free_nodes):
                continue
            # Find the existing flat node: divisor=1, length=tree.numel
            flat_node = None
            for n in free_nodes:
                if int(n.divisor) == 1 and V.graph.sizevars.statically_known_equals(
                    n.length, tree.numel
                ):
                    flat_node = n
                    break
            if flat_node is None:
                continue
            others = [n for n in free_nodes if n is not flat_node]
            if len(others) < 1:
                continue
            # Verify others form a contiguous row-major chain:
            # sorted by divisor, divisor[0]=1, divisor[i+1]=divisor[i]*length[i]
            others.sort(key=lambda n: int(n.divisor))
            if int(others[0].divisor) != 1:
                continue
            chain_ok = True
            expected_div = 1
            for n in others:
                if int(n.divisor) != expected_div:
                    chain_ok = False
                    break
                if isinstance(n.length, (int, sympy.Integer)):
                    expected_div *= int(n.length)
                else:
                    # Dynamic length: can't verify next divisor statically,
                    # but if this is the last node in the chain it's fine
                    expected_div = None
                    break
            if not chain_ok:
                continue
            # Verify product of others' lengths equals tree.numel
            if expected_div is not None:
                if not V.graph.sizevars.statically_known_equals(
                    sympy.Integer(expected_div), tree.numel
                ):
                    continue
            else:
                # Last node has dynamic length: verify symbolically
                product = sympy.Integer(1)
                for n in others:
                    product = product * n.length
                if not V.graph.sizevars.statically_known_equals(product, tree.numel):
                    continue
            # Map others to expressions of flat_node.
            # The outermost node (last in chain, highest divisor) doesn't need
            # a modulo — flat_sym // d is already < L since flat_sym < numel
            # = d * L. This avoids referencing a dynamic numel arg that won't
            # be emitted (mapped nodes don't get numel args).
            flat_sym = flat_node.symbol()
            for idx, n in enumerate(others):
                d = int(n.divisor)
                is_outermost = (idx == len(others) - 1)
                if isinstance(n.length, (int, sympy.Integer)):
                    L = int(n.length)
                    if d == 1:
                        text = f"({flat_sym} % {L})"
                    elif is_outermost:
                        text = f"({flat_sym} // {d})"
                    else:
                        text = f"(({flat_sym} // {d}) % {L})"
                else:
                    # Dynamic length on outermost: just floor-div, no mod needed
                    if is_outermost:
                        text = str(flat_sym) if d == 1 else f"({flat_sym} // {d})"
                    else:
                        L_str = f"{n.name}numel"
                        text = f"({flat_sym} % {L_str})" if d == 1 else f"(({flat_sym} // {d}) % {L_str})"
                tree.tree_node_mapping[n.name] = _FlatMapExpr(
                    text, [flat_sym]
                )
                pattern = f"{n.name} = {n.name}index"
                matcher[pattern] = f"{n.name} = {text}"

    @staticmethod
    def _rewrite_flat_r_loop_inplace(kernel, r_info):
        """Rewrite flat reduction loops already in kernel.body into nested loops.

        When a fused kernel has multiple reduction passes (e.g. mean then
        variance), the first pass is flushed into kernel.body before
        linearize_info is set. This method rewrites those flat loops in-place.
        """
        tree = r_info['tree']
        inner_node = r_info['inner_node']
        inner_len = r_info['inner_len']
        outer_nodes = r_info['outer_nodes']
        prefix = tree.prefix
        inner_name = inner_node.name

        def _len_str(length, node):
            if isinstance(length, (int, sympy.Integer)):
                return str(int(length))
            return f"{node.name}numel"

        # Check if any flat r-loop exists in body
        has_flat_loop = any(
            isinstance(line, str) and _is_flat_rloop_header(line.strip(), prefix)
            and f"{prefix}numel" in line
            for line in kernel.body._lines
        )
        if not has_flat_loop:
            return

        # Check if nested loops already exist
        has_outer_loop = any(
            isinstance(line, str) and any(
                f"for {onode.name} in range(" in line for onode in outer_nodes
            )
            for line in kernel.body._lines
        )
        if has_outer_loop:
            return

        inner_len_str = _len_str(inner_len, inner_node)
        extra_indent = "    " * len(outer_nodes)

        new_lines = []
        inside_r_loop = False
        loop_indent = ""
        for line in kernel.body._lines:
            if not isinstance(line, str):
                if inside_r_loop:
                    from torch._inductor.codegen.common import DeferredLineBase
                    if isinstance(line, DeferredLineBase):
                        new_lines.append(line.with_prefix(extra_indent))
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
                continue
            stripped = line.strip()
            indent = line[:len(line) - len(line.lstrip())] if stripped else ""

            if not inside_r_loop and _is_flat_rloop_header(stripped, prefix):
                loop_indent = indent
                cur_indent = indent
                for onode in outer_nodes:
                    olen_str = _len_str(onode.length, onode)
                    new_lines.append(f"{cur_indent}for {onode.name} in range({olen_str}):")
                    cur_indent += "    "
                new_lines.append(f"{cur_indent}for {prefix}offset in range(0, {inner_len_str}, {prefix.upper()}BLOCK):")
                inside_r_loop = True
                continue

            if inside_r_loop:
                # Detect end of loop body: indent level back to or less than loop_indent
                if stripped and not indent.startswith(loop_indent + "    "):
                    inside_r_loop = False
                    new_lines.append(line)
                    continue
                if f"{prefix}mask = {prefix}index < " in stripped:
                    new_lines.append(f"{indent}{extra_indent}{prefix}mask = {prefix}index < {inner_len_str}")
                    continue
                if f"{inner_name} = " in stripped and "%" in stripped:
                    new_lines.append(f"{indent}{extra_indent}{inner_name} = {prefix}index")
                    continue
                # Rewrite a folded flat reduction node assignment
                # (`r0_3 = r0_index`) into its sub-axis decomposition so loads
                # indexing by the flat node stay valid (see _apply_linearize's
                # reduction flat-node fold).
                _flat_subs = getattr(kernel, "_npu_flat_rnode_subs", None)
                if _flat_subs:
                    _fname = stripped.split(" = ", 1)[0] if " = " in stripped else None
                    if _fname in _flat_subs:
                        new_lines.append(f"{indent}{extra_indent}{_fname} = {_flat_subs[_fname]}")
                        continue
                skip = False
                for onode in outer_nodes:
                    if f"{onode.name} = " in stripped and "//" in stripped:
                        skip = True
                        break
                if skip:
                    continue
                new_lines.append(f"{extra_indent}{line}")
                continue

            new_lines.append(line)

        kernel.body._lines = new_lines

    @staticmethod
    def _npu_order_trees_by_stride(kernel):
        """Return range_trees ordered by innermost memory stride descending.

        Outer slot (slot 0) → tree with the largest innermost memory stride.
        Inner slot (last) → tree with the smallest (typically stride-1).

        Default upstream walks range_trees in declaration order (X first,
        R last), which always puts R innermost — correct for last-axis
        reductions (rindex stride 1) but wrong for leading-axis reductions
        like sum(dim=0) on [s0, s1, s2], where rindex stride = s1*s2 ≫ xindex
        stride 1. Innermost-R there forces every R step to a stride-(s1*s2)
        DMA and kills MTE coalescing.

        Stride source: the actual loads/stores in the schedule. For each
        scheduler node we ask `kernel.get_strides_of_load(dep.index)` which
        returns {xindex_sym: stride, rindex_sym: stride}; we take the min
        over all deps per tree (most-contiguous stride wins). Falls back
        to declaration order on any introspection failure or when no deps
        expose every tree.
        """
        trees = list(kernel.range_trees)
        if len(trees) <= 1:
            return trees
        try:
            from torch._inductor.dependencies import MemoryDep
            # Two-pass aggregation. joint_max[tree] — max stride over deps where BOTH a
            #   pointwise and reduction var appear (only these disclose pw-vs-red layout);
            #   max not min so the heaviest tensor wins. fallback_min[tree] — min stride
            #   over all deps, used only when no joint dep yields strides (still must
            #   return something in declaration order).
            joint_max: Dict[int, int] = {}
            fallback_min: Dict[int, int] = {}
            sched_nodes = []
            features = getattr(kernel, "features", None)
            if features is not None and hasattr(features, "scheduler_nodes"):
                sched_nodes = list(features.scheduler_nodes())
            for node in sched_nodes:
                rw = getattr(node, "read_writes", None)
                if rw is None:
                    continue
                # Split point: number of pointwise iter vars. _sizes is
                # (pointwise_dims, reduction_dims). var_names follow the
                # same order: pointwise vars first, then reduction vars.
                sizes = getattr(node, "_sizes", None)
                if sizes is None or len(sizes) < 2:
                    continue
                pw_count = len(sizes[0])
                deps = list(rw.reads) + list(rw.writes)
                for dep in deps:
                    if not isinstance(dep, MemoryDep):
                        continue
                    if len(dep.var_names) < pw_count:
                        continue
                    try:
                        strides = V.graph.sizevars.stride_hints(dep.index, dep.var_names)
                    except Exception:
                        continue
                    # Per-tree stride: min of |stride| over the vars whose
                    # corresponding dim length > 1 (broadcast dims have
                    # stride 0 and would otherwise dominate the min).
                    pw_strides = []
                    red_strides = []
                    for idx, (s, sz) in enumerate(zip(strides, dep.size)):
                        try:
                            sz_val = int(sz) if isinstance(sz, (int, sympy.Integer)) else int(V.graph.sizevars.optimization_hint(sz))  # noqa: B950
                            s_val = int(s) if isinstance(s, (int, sympy.Integer)) else int(V.graph.sizevars.optimization_hint(s))
                        except Exception:
                            continue
                        if sz_val <= 1 or s_val <= 0:
                            continue
                        if idx < pw_count:
                            pw_strides.append(s_val)
                        else:
                            red_strides.append(s_val)
                    is_joint = bool(pw_strides) and bool(red_strides)
                    for tree in trees:
                        relevant = red_strides if tree.is_reduction else pw_strides
                        if not relevant:
                            continue
                        cand = min(relevant)
                        # Joint-dep aggregate: max across deps (heaviest
                        # tensor's layout wins, broadcasts can't override).
                        if is_joint:
                            prev = joint_max.get(id(tree))
                            if prev is None or cand > prev:
                                joint_max[id(tree)] = cand
                        # Fallback min: classic safety net when no joint
                        # dep is available.
                        prev_fb = fallback_min.get(id(tree))
                        if prev_fb is None or cand < prev_fb:
                            fallback_min[id(tree)] = cand
            # Prefer joint-dep verdict when it covers all trees.
            if all(id(t) in joint_max for t in trees):
                tree_stride = joint_max
            elif all(id(t) in fallback_min for t in trees):
                tree_stride = fallback_min
            else:
                return trees
            result = sorted(
                trees,
                key=lambda t: tree_stride.get(id(t), 0),
                reverse=True,
            )
            return result
        except Exception:
            return trees

    def _npu_repermute_tensor_dims(self, kernel, kernel_features):
        """Pre-body hook: permute tree.tensor_dim by memory stride order.

        Must run BEFORE codegen_node_schedule_with_kernel: body codegen
        stringifies dense_size_str() at emit time (via tl.full([..]),
        tl.broadcast_to(.., [..]), tl.sum(_tmp, dim)), so any change after
        that leaves stale shapes in body and produces wrong results. The
        complementary post-body re-anchor of var_tensor_dims happens in
        _apply_linearize, which now iterates trees by tensor_dim ascending.
        """
        if kernel is None:
            return
        if not getattr(kernel, "inside_reduction", False):
            return
        ordered = self._npu_order_trees_by_stride(kernel)
        if [id(t) for t in ordered] == [id(t) for t in kernel.range_trees]:
            return
        slot = 0
        for tree in ordered:
            if tree.tensor_dim is None:
                continue
            tree.tensor_dim = slot
            slot += 1
        # Mark kernel permuted so _apply_linearize iterates trees by tensor_dim ascending,
        # not declaration order. Un-permuted kernels (pointwise) have upstream tensor_dim
        # unaligned with declaration order (X.tensor_dim=0 < Y.tensor_dim=1 though
        # range_trees is [Y, X]); sorting there would flip the slot layout and break the
        # contiguous-axis-innermost invariant the divisor sort sets up.
        kernel._npu_tile_permuted = True

    def _fold_dual_decomp(self, kernel, tree, tree_expr, tree_node_mapping, matcher):
        """Collapse a secondary full divisor-chain decomposition onto the basis.

        See the call site (NPU_FOLD_DUAL_DECOMP) for the data-layout rationale.
        Maps every node of the *other* complete chain to div/mod of the basis
        flat index (added to tree_node_mapping + matcher so it stops being an
        independent axis), and registers an address-text substitution that
        rewrites the other chain's flat-reconstruction (e.g. `x2 + 16384*x3`)
        back to the basis flat text (`x0 + 128*x1`) inside emitted load/store
        addresses — keeping the access a contiguous burst.
        """
        sizevars = V.graph.sizevars
        free_nodes = [n for n in tree.nodes.values()
                      if n.name not in tree_node_mapping]
        basis_names = {str(v) for v in tree_expr[0]}
        basis_chain = [n for n in free_nodes if n.name in basis_names]
        other_chain = [n for n in free_nodes if n.name not in basis_names]

        def _complete_chain(nodes):
            """Return divisor-sorted nodes iff they form a contiguous chain
            covering [0, numel): divisor[0]==1, divisor[i+1]==divisor[i]*len[i],
            product==numel. Else None."""
            if not nodes:
                return None
            ns = sorted(nodes, key=lambda n: sizevars.optimization_hint(n.divisor))
            if not (isinstance(ns[0].divisor, (int, sympy.Integer))
                    and int(ns[0].divisor) == 1):
                return None
            expected = sympy.Integer(1)
            for o in ns:
                if not sizevars.statically_known_equals(o.divisor, expected):
                    return None
                expected = o.divisor * o.length
            if not sizevars.statically_known_equals(expected, tree.numel):
                return None
            return ns

        basis_sorted = _complete_chain(basis_chain)
        other_sorted = _complete_chain(other_chain)
        # Only fold a genuine dual decomposition: the basis covers the whole
        # space on its own AND a second disjoint chain also covers it.
        if not (basis_sorted and other_sorted):
            return

        def _chain_flat(nodes):
            terms = []
            for n in nodes:
                sym = sympy.Symbol(n.name)
                if isinstance(n.divisor, (int, sympy.Integer)) and int(n.divisor) == 1:
                    terms.append(sym)
                else:
                    terms.append(sym * n.divisor)
            return sum(terms[1:], terms[0])

        flat_expr = _chain_flat(basis_sorted)
        recon_expr = _chain_flat(other_sorted)
        flat_text = texpr(flat_expr)
        recon_text = texpr(recon_expr)

        # Map each secondary node to div/mod of the basis flat index.
        for o in other_sorted:
            is_top = sizevars.statically_known_equals(o.divisor * o.length, tree.numel)
            if is_top:
                node_expr = flat_expr if isinstance(o.divisor, (int, sympy.Integer)) and int(o.divisor) == 1 else FloorDiv(flat_expr, o.divisor)  # noqa: B950
            else:
                node_expr = ModularIndexing(flat_expr, o.divisor, o.length)
            tree_node_mapping[o.name] = node_expr
            matcher[f"{o.name} = {o.name}index"] = f"{o.name} = {node_expr}"

        # Address-text fix: rewrite the secondary chain's flat reconstruction
        # back to the basis flat text on load/store address lines, so the access
        # stays contiguous instead of degrading to a div+mod scatter.
        if recon_text != flat_text:
            addr_subs = getattr(kernel, "_npu_addr_text_subs", None)
            if addr_subs is None:
                addr_subs = {}
                kernel._npu_addr_text_subs = addr_subs
            addr_subs[recon_text] = flat_text


    @staticmethod
    def _divisor_chain_ok(others_sorted, sizevars):
        """Verify others_sorted is a contiguous divisor chain (each divisor ==
        product of previous lengths). Returns (ok, expected_div) or (False, None)."""
        expected_div = sympy.Integer(1)
        for o in others_sorted:
            if not sizevars.statically_known_equals(o.divisor, expected_div):
                if sizevars.optimization_hint(o.divisor) != sizevars.optimization_hint(expected_div):
                    return False, None
            expected_div = o.divisor * o.length
        return True, expected_div

    @staticmethod
    def _flat_node_expr(others_sorted):
        """Flat index = sum(node_i * divisor_i) over the chain (divisor==1 -> bare symbol)."""
        terms = []
        for o in others_sorted:
            sym = sympy.Symbol(o.name)
            if isinstance(o.divisor, (int, sympy.Integer)) and int(o.divisor) == 1:
                terms.append(sym)
            else:
                terms.append(sym * o.divisor)
        return sum(terms[1:], terms[0]) if terms else sympy.Integer(0)

    def _apply_linearize(self, kernel, node_schedule):
        """
        Post-process a kernel for linearize mode:
        - Build tree_node_mapping for multi-dimensional iteration
        - Remap tensor dims to per-node dims
        - Replace codegen_range_tree with NPU version
        - Replace tl.program_id(0) with group-based dispatch
        """
        if kernel is None:
            return

        def indexer(index_var, var_range):
            strides = [sympy.Integer(1)]
            for dim in reversed(var_range[1:]):
                strides.append(strides[-1] * dim)
            strides = list(reversed(strides))
            if not (len(index_var) == len(strides) == len(var_range)):
                raise RuntimeError('Not len(index) == len(strides) == len(var_range)!')
            result = sympy.Integer(0)
            for idx, stride, sz in zip(index_var, strides, var_range):
                if sz != 1:
                    result = result + idx * stride
            return result

        matcher = {}

        kernel.index_vars_per_node = [
            [
                [item for item in sublist if not str(item).startswith('r')]
                for sublist in nested_list
            ]
            for nested_list in kernel.index_vars_per_node
        ]
        kernel.var_ranges_per_node = [
            [
                [item2 for item1, item2 in zip(sublist1, sublist2) if not str(item1).startswith('r')]
                for sublist1, sublist2 in zip(nested_list1, nested_list2)
            ]
            for nested_list1, nested_list2 in zip(kernel.index_vars_per_node, kernel.var_ranges_per_node)
        ]

        for i, tree in enumerate(kernel.range_trees):
            tree_expr = None
            # Preserve mappings already inserted by NPUTritonKernel.prepare_indexing
            # (e.g. fused-axis splits) — they were registered while load/store
            # codegen ran and must not be wiped by the rebuild below.
            tree_node_mapping = dict(getattr(tree, 'tree_node_mapping', {}) or {})
            max_rank = 0

            for id, (var_ranges, index_vars) in enumerate(
                sorted(
                    zip(kernel.var_ranges_per_node, kernel.index_vars_per_node),
                    key=lambda pair: len(pair[0][i]) if i < len(pair[0]) else 0,
                    reverse=True,
                )
            ):
                if i >= len(var_ranges) or i >= len(index_vars):
                    continue
                var_range = var_ranges[i]
                index_var = index_vars[i]
                assert len(var_range) == len(index_var)
                if len(var_range) == 0:
                    continue
                if id == 0:
                    tree_expr = (index_var, var_range)
                    max_rank = len(var_range)
                    continue
                elif len(var_range) < max_rank:
                    start = 0
                    for index_i, size_i in zip(index_var, var_range):
                        length = sympy.Integer(1)
                        name = str(index_i)
                        for ind in range(start, len(tree_expr[0])):
                            length *= tree_expr[1][ind]
                            if V.graph.sizevars.statically_known_equals(length, size_i):
                                if start == ind:
                                    start = ind + 1
                                    length = sympy.Integer(1)
                                    break
                                node_expr = indexer(tree_expr[0][start:ind + 1], tree_expr[1][start:ind + 1])
                                tree_node_mapping[name] = node_expr
                                pattern = f"{name} = {name}index"
                                replacement = f"{name} = {node_expr}"
                                matcher[pattern] = replacement
                                length = sympy.Integer(1)
                                start = ind + 1
                                break
                elif (
                    len(var_range) == max_rank
                    and ncfg.fold_transposed_xnode
                ):
                    # Transposed dual-view: same rank as basis tree_expr but axes are a
                    # pure PERMUTATION (same lengths reordered) — stores [50,4] vs [4,50]
                    # over 200 tasks. Unfolded, x0,x1 and x3,x4 are 4 independent x-axes →
                    # Cartesian grid (40000 vs 200, 200x). Map each axis onto the basis
                    # axis of SAME length; skip if any length is duplicated (ambiguous).
                    sv = V.graph.sizevars
                    basis_vars, basis_ranges = tree_expr
                    # Lengths must be a permutation of the basis lengths.
                    used = [False] * len(basis_ranges)
                    pairing = {}  # this-node index_var -> basis index_var
                    ok = len(var_range) == len(basis_ranges)
                    for index_i, size_i in zip(index_var, var_range):
                        match_j = None
                        ambiguous = False
                        for j, bsz in enumerate(basis_ranges):
                            if used[j]:
                                continue
                            if sv.optimization_hint(bsz) == sv.optimization_hint(size_i) and sv.statically_known_equals(bsz, size_i):  # noqa: B950
                                if match_j is not None:
                                    ambiguous = True
                                    break
                                match_j = j
                        if match_j is None or ambiguous:
                            ok = False
                            break
                        used[match_j] = True
                        pairing[str(index_i)] = basis_vars[match_j]
                    if ok and str(index_var) != str(basis_vars):
                        for name, basis_sym in pairing.items():
                            tree_node_mapping[name] = basis_sym
                            matcher[f"{name} = {name}index"] = f"{name} = {basis_sym}"

            # Second pass: detect flattened auxiliary nodes.
            # Pattern: one node has divisor=1 and length==tree.numel (the flat
            # view of the entire iteration space). The other nodes decompose
            # that space with a divisor chain. Map the flat node to:
            #   flat = sum(node_i * divisor_i) for each decomposed node_i.
            if tree_expr is not None and not tree.is_reduction:
                free_nodes = [n for n in tree.nodes.values()
                              if n.name not in tree_node_mapping]
                if len(free_nodes) > 2:
                    sizevars = V.graph.sizevars
                    numel_hint = sizevars.optimization_hint(tree.numel)
                    flat_node = None
                    for n in free_nodes:
                        if isinstance(n.divisor, (int, sympy.Integer)) and int(n.divisor) == 1:
                            if sizevars.optimization_hint(n.length) == numel_hint:
                                if sizevars.statically_known_equals(n.length, tree.numel):
                                    flat_node = n
                                    break
                    if flat_node is None:
                        for n in free_nodes:
                            if isinstance(n.divisor, (int, sympy.Integer)) and int(n.divisor) == 1:
                                if sizevars.optimization_hint(n.length) == numel_hint:
                                    flat_node = n
                                    break
                    if flat_node is not None:
                        others = [n for n in free_nodes if n is not flat_node]
                        others_sorted = sorted(others, key=lambda n: sizevars.optimization_hint(n.divisor))
                        if isinstance(others_sorted[0].divisor, (int, sympy.Integer)) and int(others_sorted[0].divisor) == 1:
                            chain_ok, _ = self._divisor_chain_ok(others_sorted, sizevars)
                            if chain_ok:
                                node_expr = self._flat_node_expr(others_sorted)
                                tree_node_mapping[flat_node.name] = node_expr
                                pattern = f"{flat_node.name} = {flat_node.name}index"
                                replacement = f"{flat_node.name} = {node_expr}"
                                matcher[pattern] = replacement

            # Third pass: dual full-decomposition fold (fold_dual_decomp). A permute
            # makes one flat space carry TWO complete divisor-chains (load decomp A,
            # store decomp B) so x0..x3 become independent grid axes (Cartesian blowup,
            # wrong). Fold: keep A as tiled axes, express every B node as div/mod of A's
            # flat index (algebraically ==flat) so the store stays a contiguous burst.
            if (
                tree_expr is not None
                and not tree.is_reduction
                and ncfg.fold_dual_decomp
            ):
                self._fold_dual_decomp(
                    kernel, tree, tree_expr, tree_node_mapping, matcher,
                )

            # Reduction-tree variant of the flat-node fold. Two reductions share the flat
            # space but index differently — 2D sub-axes (r0_1,r0_2) vs a flat node (r0_3,
            # length==numel); leaving r0_3 makes codegen emit `for r0_3: for r0_2`, re-run
            # the inner reduction numel× (slow AND wrong). Map it onto the sub-axis decomp
            # to exclude from outer_nodes. Gated separately; x-axis stays byte-identical.
            if (
                tree.is_reduction
                and ncfg.fold_flat_rnode
            ):
                free_nodes = [n for n in tree.nodes.values()
                              if n.name not in tree_node_mapping]
                if len(free_nodes) >= 2:
                    sizevars = V.graph.sizevars
                    numel_hint = sizevars.optimization_hint(tree.numel)
                    flat_node = None
                    for n in free_nodes:
                        if (isinstance(n.divisor, (int, sympy.Integer)) and int(n.divisor) == 1
                                and sizevars.optimization_hint(n.length) == numel_hint
                                and sizevars.statically_known_equals(n.length, tree.numel)):
                            flat_node = n
                            break
                    # Only fold when there is at least one *other* node that
                    # decomposes the same space (otherwise the flat node IS the
                    # sole reduction axis and must stay).
                    others = [n for n in free_nodes if n is not flat_node]
                    if flat_node is not None and others:
                        others_sorted = sorted(others, key=lambda n: sizevars.optimization_hint(n.divisor))
                        if isinstance(others_sorted[0].divisor, (int, sympy.Integer)) and int(others_sorted[0].divisor) == 1:
                            chain_ok, expected_div = self._divisor_chain_ok(others_sorted, sizevars)
                            # The decomposition must cover the whole flat space.
                            if chain_ok and not sizevars.statically_known_equals(expected_div, tree.numel):
                                chain_ok = False
                            if chain_ok:
                                node_expr = self._flat_node_expr(others_sorted)
                                tree_node_mapping[flat_node.name] = node_expr
                                # Flat node is assigned `r0_3 = r0_index` (prefix-level),
                                # so record the sub for codegen_body to rewrite into
                                # `r0_3 = <decomp expr>`. Render via rename_indexing so
                                # symbolic divisors map to arg names (ks0), else str()
                                # yields the raw size symbol (s70) → NameError.
                                flat_subs = getattr(kernel, "_npu_flat_rnode_subs", None)
                                if flat_subs is None:
                                    flat_subs = {}
                                    kernel._npu_flat_rnode_subs = flat_subs
                                renamed_expr = kernel.rename_indexing(node_expr)
                                flat_subs[flat_node.name] = kernel.kexpr(renamed_expr)
                                # Tag as a split/fold RECONSTRUCTION alias so rtree
                                # promotion allows this tree (mapped node rebuilds from
                                # free sub-node aranges, not deleted r0_index); else the
                                # mapping-guard rejects it into nested scalar loops. The
                                # dual-decomp guard still gates unpromotable shapes.
                                recon_nodes = getattr(tree, "_npu_split_recon_nodes", None)
                                if recon_nodes is None:
                                    recon_nodes = set()
                                    tree._npu_split_recon_nodes = recon_nodes
                                recon_nodes.add(flat_node.name)

            tree.tree_node_mapping = tree_node_mapping

        # Expand FloorDiv/Mod patterns on single-node trees into sub-nodes.
        if ncfg.expand_divmod:
            with V.set_kernel_handler(kernel):
                self._expand_divmod_nodes(kernel, matcher)

        # Collapse contiguous row-major x-tree nodes into one flat node. Free nodes in a
        # contiguous row-major layout (product==numel) are really a 1D iteration; keeping
        # them 3D makes the autotuner pick lane-wasting XBLOCK splits. Fold to one flat F
        # (old nodes = xi=(F//d)%L) so a 1D tile spans xnumel evenly. Skip pointwise: the
        # collapse adds modular address arithmetic, slower than the natural multi-dim.
        if ncfg.collapse_xtree and kernel.inside_reduction:
            with V.set_kernel_handler(kernel):
                self._collapse_rowmajor_xtrees(kernel, matcher)

        # R-tree real-block promotion (rtree_real_block): per tree, decide if free sub-nodes
        # each become a REAL tile dim (real_block_rN) vs nested scalar loops → multi-dim
        # accumulator, reshape-collapsed before one tl.sum. Decided HERE (after folds) so
        # Sites A/B/C/E/F read the marker. Default OFF (byte-identical); can't default ON:
        # miscomputes clip_qkv_bias_grad_sum (wrong).
        kernel._npu_rtree_real_block = ncfg.rtree_real_block
        # tree.prefix -> ordered free-node names (divisor DESCENDING: stride-large
        # outer slot first, contiguous stride-1 node innermost). Empty/absent =>
        # not promoted, legacy nested-loop path.
        kernel._npu_rtree_promoted = {}
        # Parallel maps: prefix -> bool (True = dynamic tile-loop form, False =
        # static fully-resident arange), node.name -> int constexpr tile width.
        # Empty/absent => not promoted (static-resident is the False default when
        # a prefix IS in _npu_rtree_promoted). Both are .get()-guarded everywhere
        # downstream so the gate-OFF / non-promoted path is byte-identical.
        kernel._npu_rtree_dynamic = {}
        kernel._npu_rtree_tile = {}
        # node.name -> bool. True = symbolic length → runtime tile-loop + {nm}numel arg;
        # False = static, fully-resident arange (no loop/arg). Per-NODE, so a dynamic
        # tree can hold static nodes (ViT: r0_1=197 seq + r0_2=N batch) that must NOT be
        # looped. Only populated for promoted trees; .get()-guarded below.
        kernel._npu_rtree_node_dynamic = {}
        if kernel._npu_rtree_real_block and kernel.inside_reduction:
            sv = V.graph.sizevars

            def _rdiv_key(n):
                if isinstance(n.divisor, (int, sympy.Integer)):
                    return int(n.divisor)
                return int(sv.optimization_hint(n.divisor))

            R_TILE_CAP = ncfg.rtree_real_block_cap
            # Kept-axis (x-tree) numel hint — the upper bound on the resident
            # block width that multiplies the reduction tile for the Triton 1M
            # tensor-numel cap. Shared by static + dynamic branches.
            x_numel = 1
            for xt in kernel.range_trees:
                if xt.is_reduction or xt.tensor_dim is None:
                    continue
                try:
                    x_numel *= int(sv.optimization_hint(xt.numel))
                except Exception:
                    x_numel = 1 << 60
                    break
            for tree in kernel.range_trees:
                if not tree.is_reduction or tree.tensor_dim is None:
                    continue
                mapping = getattr(tree, "tree_node_mapping", {}) or {}
                free_r = [n for n in tree.nodes.values()
                          if n.name not in mapping]
                if len(free_r) < 2:
                    continue
                # Mapped-node (dual-VIEW) guard: promotion deletes the flat r0_index
                # scaffolding, so a mapped alias ``r0_3 = r0_index`` dangles → NameError.
                # EXCEPTION: split RECONSTRUCTION aliases (``r0_1 = r0_2 + ks0*r0_3``)
                # reference free sub-nodes promotion DOES emit, so they reconstruct
                # inside the body. Allow when every mapped node is such a recon.
                recon_nodes = getattr(tree, "_npu_split_recon_nodes", None) or set()
                if mapping and not all(nm in recon_nodes for nm in mapping):
                    continue
                free_sorted = sorted(free_r, key=_rdiv_key, reverse=True)
                is_dynamic = any(
                    not isinstance(n.length, (int, sympy.Integer))
                    for n in free_r
                )
                # Dual-decomp guard: only promote a CLEAN single decomposition where
                # free r-nodes are independent axes whose ∏length == flat numel. Two
                # fused reductions viewing the SAME space through DIFFERENT decomps
                # (dcgan bwd: r0_1=H*W,r0_2=N vs r0_3=H,r0_4=W,r0_2=N) overshoot numel;
                # promoting sums subset-loads N_other× (wrong). Leave to legacy folds.
                try:
                    _len_prod = 1
                    for n in free_r:
                        _len_prod *= int(sv.optimization_hint(n.length))
                    _numel_hint = int(sv.optimization_hint(tree.numel))
                except Exception:
                    continue
                if _len_prod != _numel_hint:
                    continue
                if not is_dynamic:
                    # Static: fully-resident arange tile (no loop). Requires the
                    # whole reduction product to fit one tile (<= R_TILE_CAP) and
                    # the kept-axis*reduction tensor numel under the Triton cap.
                    lengths = [int(n.length) for n in free_r]
                    r_product = 1
                    for L in lengths:
                        r_product *= L
                    if r_product <= R_TILE_CAP and x_numel * r_product <= 1_000_000:
                        kernel._npu_rtree_promoted[tree.prefix] = [
                            n.name for n in free_sorted
                        ]
                        kernel._npu_rtree_dynamic[tree.prefix] = False
                        for n in free_r:
                            kernel._npu_rtree_tile[n.name] = int(n.length)
                            kernel._npu_rtree_node_dynamic[n.name] = False
                        continue
                    # Over-cap static: can't be fully resident. Tile like the dynamic
                    # path — inner nodes resident up to R_TILE_CAP, outer node(s) get a
                    # real-block tile-LOOP with literal bounds (r0_2=77 @64 → 2 iters).
                    # Static lengths keep bounds/masks literal, no {nm}numel arg.
                    inner_first = sorted(free_r, key=_rdiv_key)  # ascending div
                    tiles = {}
                    node_looped = {}
                    inner_prod = 1
                    for n in inner_first:
                        L = int(n.length)
                        if inner_prod * L <= R_TILE_CAP:
                            # Fully resident: whole node fits the remaining budget.
                            tiles[n.name] = L
                            node_looped[n.name] = False
                            inner_prod *= L
                        else:
                            # Tile this node: as many lanes as the budget allows
                            # (>=1), the loop covers the rest of its length.
                            tile = max(1, R_TILE_CAP // inner_prod)
                            tile = min(tile, L)
                            tiles[n.name] = tile
                            node_looped[n.name] = True
                            inner_prod *= tile
                    tile_product = 1
                    for t in tiles.values():
                        tile_product *= t
                    if x_numel * tile_product > 1_048_576:
                        continue
                    # Require at least one resident node (else it degrades to a
                    # pure scalar loop with no vectorised inner tile).
                    if all(node_looped.values()):
                        continue
                    kernel._npu_rtree_promoted[tree.prefix] = [
                        n.name for n in free_sorted
                    ]
                    kernel._npu_rtree_dynamic[tree.prefix] = True
                    for n in free_r:
                        kernel._npu_rtree_tile[n.name] = tiles[n.name]
                        kernel._npu_rtree_node_dynamic[n.name] = node_looped[n.name]
                    continue
                # Dynamic: symbolic length(s). Constexpr tile per node + a runtime
                # accumulation loop (mirrors the x-tree dynamic path); the loop covers
                # any runtime size so only per-tile numel is bounded. Greedy alloc in
                # divisor-ASCENDING order (innermost first), tile clamped to length hint.
                try:
                    length_hints = {
                        n.name: int(sv.optimization_hint(n.length))
                        for n in free_r
                    }
                except Exception:
                    continue
                if any(h <= 0 for h in length_hints.values()):
                    continue
                # Per-node static/dynamic split: static-length nodes are fully resident
                # (no loop, no {nm}numel arg), only symbolic nodes get sub-tile+loop, so
                # a dynamic tree can hold static nodes (ViT r0_1=197 seq + r0_2=N dyn).
                node_is_dyn = {
                    n.name: not isinstance(n.length, (int, sympy.Integer))
                    for n in free_r
                }
                # Resident (static) nodes consume their full length up front; the
                # dynamic budget is what remains under R_TILE_CAP after them.
                static_prod = 1
                for n in free_r:
                    if not node_is_dyn[n.name]:
                        static_prod *= length_hints[n.name]
                if static_prod <= 0 or static_prod > R_TILE_CAP:
                    continue
                inner_first = sorted(free_r, key=_rdiv_key)  # ascending divisor
                tiles = {}
                inner_prod = static_prod
                ok = True
                for n in inner_first:
                    lh = length_hints[n.name]
                    if not node_is_dyn[n.name]:
                        # Fully resident: tile == full static length.
                        tiles[n.name] = lh
                        continue
                    tile = min(lh, max(1, R_TILE_CAP // inner_prod))
                    if tile <= 0:
                        ok = False
                        break
                    tiles[n.name] = tile
                    inner_prod *= tile
                if not ok:
                    continue
                tile_product = 1
                for t in tiles.values():
                    tile_product *= t
                if x_numel * tile_product > 1_048_576:
                    continue
                kernel._npu_rtree_promoted[tree.prefix] = [
                    n.name for n in free_sorted
                ]
                kernel._npu_rtree_dynamic[tree.prefix] = True
                for n in free_r:
                    kernel._npu_rtree_tile[n.name] = tiles[n.name]
                    kernel._npu_rtree_node_dynamic[n.name] = node_is_dyn[n.name]

        # Compute old_dense_size BEFORE tensor_dim mutation (matches codegen_body output).
        orig_ndim = sum(int(tree.tensor_dim is not None) for tree in kernel.range_trees)
        old_sizes = ["1"] * orig_ndim
        for tree in kernel.range_trees:
            if tree.tensor_dim is None:
                continue
            if not tree.is_reduction or kernel.inside_reduction:
                old_sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"
        old_dense_size = f"[{', '.join(old_sizes)}]"
        old_reduction_dim = orig_ndim - 1 if not kernel.no_x_dim else 1
        old_slice = "[" + ", ".join([":"] * (orig_ndim - 1) + ["None"]) + "]"

        # Cross-tree slot order. Deviate from declaration order only when the hook
        # permuted the trees (_npu_tile_permuted). Upstream's tensor_dim doesn't match
        # range_trees order (built Y-before-X, but X.tensor_dim=0/Y=1), so for un-permuted
        # kernels keep the declaration-order walk — sorting by tensor_dim would flip Y/X
        # slots and break MTE coalescing.
        slot_order = sorted(
            kernel.range_trees,
            key=lambda t: t.tensor_dim if t.tensor_dim is not None else len(kernel.range_trees),
        ) if getattr(kernel, "_npu_tile_permuted", False) else list(kernel.range_trees)

        r_tensor_dim = 0
        for tree in slot_order:
            tree.var_tensor_dims = dict()
            if not tree.is_reduction and not kernel.no_x_dim:
                # Slot assignment: smallest divisor (stride-1 axis) takes the innermost
                # slot so tl.arange[...,:] sits on the contiguous axis (single aligned
                # MTE burst per row). Insertion order is unreliable (fx may insert axes
                # stride-permuted), so sort free nodes by divisor descending: largest ->
                # slot 0 (outermost), smallest -> last slot (innermost).
                free_nodes = [
                    node for node in tree.nodes.values()
                    if node.name not in tree.tree_node_mapping
                ]

                def _divisor_sort_key(n):
                    if isinstance(n.divisor, (int, sympy.Integer)):
                        return int(n.divisor)
                    return int(V.graph.sizevars.optimization_hint(n.divisor))

                free_nodes.sort(key=_divisor_sort_key, reverse=True)
                # Scalar odometer axes (tile-1 strided, behind a contiguous burst) hold
                # no register tensor-dim slot; recorded on the tree so header codegen
                # emits them as offset-only scalars and ndim excludes them.
                _scalar_odo = _npu_scalar_odometer_axis_names(kernel, tree)
                tree._npu_scalar_odometer = _scalar_odo
                _tiled_nodes = [n for n in free_nodes if n.name not in _scalar_odo]
                if _tiled_nodes and tree.tensor_dim is not None:
                    # Anchor the X-tree primary slot at its first TILED sub-node
                    # so dense_size_str() emits XBLOCK at the correct slot
                    # when tree iteration order has been re-shuffled.
                    tree.tensor_dim = r_tensor_dim
                for node in free_nodes:
                    if node.name in _scalar_odo:
                        continue  # scalar odometer axis: no register slot
                    tree.var_tensor_dims[node.name] = r_tensor_dim
                    r_tensor_dim += 1
            elif (
                tree.is_reduction
                and tree.tensor_dim is not None
                and tree.prefix in getattr(kernel, "_npu_rtree_promoted", {})
            ):
                # Promoted reduction tree: each free sub-node becomes its own real tile
                # dim (like the X-tree), not a single R0_BLOCK + nested scalar loops.
                # Slot order = cached divisor-descending (stride-1 node innermost) so the
                # inner tl.arange lands on the stride-1 axis; tensor_dim anchors the first
                # r-slot so r_tree_slot discovery finds the reduction start.
                ordered_names = kernel._npu_rtree_promoted[tree.prefix]
                tree.tensor_dim = r_tensor_dim
                for nm in ordered_names:
                    tree.var_tensor_dims[nm] = r_tensor_dim
                    r_tensor_dim += 1
            elif tree.tensor_dim is not None:
                tree.tensor_dim = r_tensor_dim
                # Bump r_tensor_dim only when the hook permuted the trees (a reduction
                # tree may sit at an inner slot with X-tree slots after it, needing
                # room). Un-permuted kernels (R last) don't need it; no-bump keeps them
                # byte-identical.
                if getattr(kernel, "_npu_tile_permuted", False):
                    r_tensor_dim += 1

        kernel._linearize_applied = True

        lines = []
        for i, line in enumerate(kernel.body._lines):
            if line not in matcher:
                lines.append(line)
        kernel.body._lines = lines

        # Reset body and regenerate the NPU range tree header, preserving ALL existing
        # body lines (post-matcher): first-pass reduction loops and post_loop_combine
        # results flushed into body by disable_reduction(). Saving only tl.full lines
        # discarded those, causing NameError for reduction results in the second pass.
        saved_body_lines = list(kernel.body._lines)
        kernel.body = IndentedBuffer()
        # Call codegen_range_tree (uses NPU-overridden iteration_ranges_get_pid
        # to emit linearize-mode pid: (group_base + i) instead of tl.program_id(0))
        kernel.codegen_range_tree()
        # Re-inject saved body lines after the header, filtering out stale r-tree base
        # arange lines (rbase/r0_base) emitted by the first pass with the old 2D shape
        # [None, :]; the correct shape is re-emitted by the call above.
        reduction_base_names = [f"{prefix}base" for prefix in kernel.get_reduction_prefixes()]
        stale_rbase_prefixes = tuple(
            f"{tree.prefix}base = tl.arange"
            for tree in kernel.range_trees
            if tree.is_reduction
        )
        stale_rbase_prefixes += tuple(
            f"rbase = {tree.prefix}base"
            for tree in kernel.range_trees
            if tree.is_reduction
        )
        stale_rbase_prefixes += tuple(
            f"{base_name} = tl.arange"
            for base_name in reduction_base_names
        )
        stale_rbase_prefixes += tuple(
            f"rbase = {base_name}"
            for base_name in reduction_base_names
        )
        for line in saved_body_lines:
            stripped = line.strip() if isinstance(line, str) else ""
            if stale_rbase_prefixes and isinstance(line, str) and stripped.startswith(stale_rbase_prefixes):
                continue
            kernel.body.writeline(line)

        # Detect r-tree split info for codegen_body override.
        # Store on kernel so codegen_body() can generate nested loops.
        if kernel.inside_reduction:
            for tree in kernel.range_trees:
                if not tree.is_reduction:
                    continue
                # Promoted r-trees use the real-block multi-tile path (per-node
                # slots already assigned above; header + sum-collapse handle the
                # rest). They have NO inner/outer split and NO nested loops, so
                # skip building _r_linearize_info entirely — codegen_body falls
                # through to the standard path for them.
                if tree.prefix in getattr(kernel, "_npu_rtree_promoted", {}):
                    continue
                all_nodes = list(tree.nodes.values())
                if len(all_nodes) < 2:
                    continue
                r_tree_node_mapping = getattr(tree, 'tree_node_mapping', {})
                free_nodes = [n for n in all_nodes if n.name not in r_tree_node_mapping]
                if len(free_nodes) < 2:
                    continue

                # Sort by divisor: smallest divisor = innermost (vectorized) node.
                # For dynamic divisors, use size_hint to determine ordering.
                def _divisor_sort_key(n):
                    if isinstance(n.divisor, (int, sympy.Integer)):
                        return int(n.divisor)
                    return int(V.graph.sizevars.optimization_hint(n.divisor))
                free_nodes_sorted = sorted(free_nodes, key=_divisor_sort_key)
                inner_node = free_nodes_sorted[0]
                outer_nodes = free_nodes_sorted[1:]


                # Dual-view sub-axis partition: an outer node whose flat span
                # (divisor*length) fits within the inner node's span re-decomposes the
                # SAME flat block; emitting it as a for loop double-counts the reduction
                # and collides int32 loop var vs int64 tensor decomp. Keep only genuine
                # outer axes (span > inner block) via optimization_hint containment.
                if outer_nodes and ncfg.fold_dualview_rnode:
                    sv = V.graph.sizevars
                    inner_span = int(sv.optimization_hint(inner_node.length))
                    outer_nodes = [
                        o for o in outer_nodes
                        if int(sv.optimization_hint(o.divisor))
                        * int(sv.optimization_hint(o.length)) > inner_span
                    ]

                # Dual-view collapse: when inner_node.length == tree.numel, outer
                # nodes are alternative decomp views of the same flat axis, not
                # independent sub-axes. Wrapping them in for loops re-runs the inner
                # loop ∏outer.length times (quadratic blowup). Drop the outer loops;
                # the per-element decomp lines in indexing_code stay valid as tensors.
                if outer_nodes:
                    try:
                        inner_eq_tree = V.graph.sizevars.statically_known_equals(
                            inner_node.length, tree.numel
                        )
                    except Exception:
                        inner_eq_tree = bool(inner_node.length == tree.numel)
                    if inner_eq_tree:
                        outer_nodes = []


                # Small outer-node collapse: flatten small outer nodes into the inner
                # vectorized loop (reduction as one flat [N×H×W] axis), avoiding nested
                # scalar loops. Heuristic: ∏outer_lengths ≤ 256 OR all lengths ≤ 8.
                # Gate flatten_small_outer_rnodes (default OFF — superseded by rtree
                # real-block promotion; kept as opt-in legacy).
                if outer_nodes and ncfg.flatten_small_outer_rnodes:
                    sv = V.graph.sizevars
                    outer_lengths = []
                    all_small = True
                    outer_product = 1
                    for onode in outer_nodes:
                        try:
                            olen = int(sv.optimization_hint(onode.length))
                            outer_lengths.append(olen)
                            outer_product *= olen
                            if olen > 8:
                                all_small = False
                        except Exception:
                            # If length is symbolic, conservatively keep the loop
                            all_small = False
                            outer_lengths.append("?")
                            outer_product = 999999  # large sentinel

                    # Flatten if: all dims ≤ 8 OR product ≤ 256
                    should_flatten = all_small or outer_product <= 256


                    if should_flatten:
                        # Flatten: treat the entire tree.numel as the inner loop
                        outer_nodes = []
                        # Update inner_len to cover the full reduction space
                        inner_len = tree.numel
                    else:
                        # Keep the nested loops
                        inner_len = inner_node.length
                else:
                    # No outer nodes or gate disabled
                    inner_len = inner_node.length
                kernel._r_linearize_info = {
                    'tree': tree,
                    'inner_node': inner_node,
                    'inner_len': inner_len,
                    'outer_nodes': outer_nodes,
                }
                kernel.linearize_info = kernel._r_linearize_info

                # Immediately rewrite any flat r-loop already flushed into
                # kernel.body (e.g. the first reduction pass in a fused
                # mean+variance kernel). codegen_body's in-place rewrite only
                # triggers when buffers are empty, but at codegen_kernel time
                # the second pass's buffers are still populated.
                self._rewrite_flat_r_loop_inplace(kernel, kernel._r_linearize_info)

        normalize_single_dim_reduction = False
        if kernel.inside_reduction and kernel.range_trees[-1].is_loop:
            new_code = IndentedBuffer()
            new_code.writeline(f"rbase = {kernel.iteration_ranges_ranges_code(kernel.range_trees[-1])}")
            new_code.splice(kernel.body)
            kernel.body._lines = new_code._lines

            reduction_prefixes = kernel.get_reduction_prefixes()
            normalized_rbase = kernel.iteration_ranges_ranges_code(kernel.range_trees[-1])
            normalize_single_dim_reduction = (
                len(reduction_prefixes) == 1 and "[None" not in normalized_rbase
            )
            if normalize_single_dim_reduction:
                prefix = reduction_prefixes[0]
                base_name = f"{prefix}base"
                filtered_lines = []
                inserted_base = False
                for line in kernel.body._lines:
                    if not isinstance(line, str):
                        filtered_lines.append(line)
                        continue
                    stripped = line.strip()
                    indent = line[:len(line) - len(line.lstrip())] if stripped else ""
                    if stripped == f"rbase = {normalized_rbase}":
                        filtered_lines.append(line)
                        if not inserted_base:
                            filtered_lines.append(f"{indent}{base_name} = {normalized_rbase}")
                            inserted_base = True
                        continue
                    if stripped.startswith(f"{base_name} = tl.arange"):
                        continue
                    filtered_lines.append(line)
                kernel.body._lines = filtered_lines

        # Compute the real linearized dense_size and reduction_dim.
        real_ndim = kernel.triton_tensor_ndim()

        # Two real_sizes strategies: permuted kernels (hook moved tensor_dim cross-
        # tree, e.g. R→0) index by tree.tensor_dim so slots match old_sizes; un-permuted
        # keep the sequential dim_idx walk (tensor_dim-indexing there mis-slots tl.full/
        # tl.broadcast_to shapes), so we deviate only when the hook moved a tree.
        real_sizes = ["1"] * real_ndim
        permuted = getattr(kernel, "_npu_tile_permuted", False)

        # Free x-sub-node slot order MUST match var_tensor_dims (divisor-descending);
        # the legacy reversed-insertion fill only coincides when insertion == divisor
        # order, else dense_size mis-slots real_block_xN and every XBLOCK>1 fails to
        # compile. _ordered_free_nodes pairs each node with its authoritative slot,
        # falling back to the legacy slot only when var_tensor_dims is unavailable.
        def _ordered_free_nodes(tree, base_slot):
            # Scalar odometer axes carry no register tile dim (real_block==1, no vtd
            # slot); drop them from the slot fill, else the all(...in vtd) check fails
            # (mis-slotting tiled axes) or a bogus slot is injected for a size-1 axis.
            _scalar_odo = getattr(tree, "_npu_scalar_odometer", None) or set()
            free = [
                n for n in tree.nodes.values()
                if n.name not in getattr(tree, 'tree_node_mapping', {})
                and n.name not in _scalar_odo
            ]
            vtd = getattr(tree, "var_tensor_dims", None) or {}
            if vtd and all(n.name in vtd for n in free):
                return [(vtd[n.name], n) for n in free]
            # Legacy fallback: reversed insertion order with incrementing slot.
            return [(base_slot + i, n) for i, n in enumerate(reversed(free))]

        if permuted:
            for tree in kernel.range_trees:
                if tree.tensor_dim is None:
                    continue
                slot = tree.tensor_dim
                if not tree.is_reduction:
                    node_block_constexpr = getattr(tree, "node_block_constexpr", {}) or {}
                    for node_slot, node in _ordered_free_nodes(tree, slot):
                        if node_slot < real_ndim:
                            real_sizes[node_slot] = node_block_constexpr.get(
                                node.name, f"real_block_{node.name}"
                            )
                elif kernel.inside_reduction:
                    if tree.prefix in getattr(kernel, "_npu_rtree_promoted", {}):
                        # Promoted r-tree: each free r-node fills its own slot with
                        # its per-node block token (not a single R0_BLOCK). Static
                        # nodes use real_block_{node}; symbolic use the constexpr tile
                        # {node}_blk. Classified per-node so a dynamic tree can hold
                        # static nodes (ViT mixed seq+batch).
                        _ndyn = getattr(kernel, "_npu_rtree_node_dynamic", {})
                        for node_slot, node in _ordered_free_nodes(tree, slot):
                            if node_slot < real_ndim:
                                real_sizes[node_slot] = (
                                    f"{node.name}_blk" if _ndyn.get(node.name)
                                    else f"real_block_{node.name}"
                                )
                    else:
                        real_sizes[slot] = f"{tree.prefix.upper()}BLOCK"
        else:
            dim_idx = 0
            for tree in kernel.range_trees:
                if tree.tensor_dim is None:
                    continue
                if not tree.is_reduction:
                    node_block_constexpr = getattr(tree, "node_block_constexpr", {}) or {}
                    for node_slot, node in _ordered_free_nodes(tree, dim_idx):
                        if node_slot < real_ndim:
                            real_sizes[node_slot] = node_block_constexpr.get(
                                node.name, f"real_block_{node.name}"
                            )
                        dim_idx = max(dim_idx, node_slot + 1)
                elif kernel.inside_reduction:
                    if tree.prefix in getattr(kernel, "_npu_rtree_promoted", {}):
                        _ndyn = getattr(kernel, "_npu_rtree_node_dynamic", {})
                        for node_slot, node in _ordered_free_nodes(tree, dim_idx):
                            if node_slot < real_ndim:
                                real_sizes[node_slot] = (
                                    f"{node.name}_blk" if _ndyn.get(node.name)
                                    else f"real_block_{node.name}"
                                )
                            dim_idx = max(dim_idx, node_slot + 1)
                    else:
                        real_sizes[dim_idx] = f"{tree.prefix.upper()}BLOCK"
                        dim_idx += 1
        real_dense_size = f"[{', '.join(real_sizes)}]"
        # R-tree's actual slot. If the hook pushed it to an outer slot, upstream's
        # tl.sum(_, ndim-nreduce) still assumes last-dim; rewrite the dim arg to match.
        # real_reduction_dim = where R lives; old_reduction_dim = what upstream emitted.
        r_tree_slot = None
        if kernel.inside_reduction:
            for t in kernel.range_trees:
                if t.is_reduction and t.tensor_dim is not None:
                    r_tree_slot = t.tensor_dim
                    break
        real_reduction_dim = 1 if kernel.no_x_dim else (r_tree_slot if r_tree_slot is not None else real_ndim - 1)

        # real_ndim can be 0 for a fully-collapsed scalar kernel (no_x_dim with
        # its single reduction reduced away — e.g. speech_transformer's
        # ``.sum()``-to-scalar inference kernels). There is no dense_size slot or
        # reduction slice to rewrite in that case, so build an empty slice and
        # let the fixup below no-op (old_dense_size won't appear as ``[]``).
        new_slice_sizes = [":"] * real_ndim
        if real_ndim == 0:
            new_slice = "[]"
        else:
            if r_tree_slot is not None and 0 <= r_tree_slot < real_ndim:
                new_slice_sizes[r_tree_slot] = "None"
            else:
                new_slice_sizes[-1] = "None"
            new_slice = "[" + ", ".join(new_slice_sizes) + "]"
        # Fire fixup whenever linearized dense_size differs, not only on ndim change:
        # even at matching ndim the slot tokens differ ([XBLOCK,R0_BLOCK] ->
        # [real_block_x0,R0_BLOCK]), and a raw XBLOCK left in tl.full/tl.broadcast_to
        # lets autotune pick an XBLOCK far larger than real_block_x0 and overrun UB.
        needs_fixup = (real_ndim != orig_ndim) or (old_dense_size != real_dense_size)

        # When any reduction tree is promoted to real-block multi-tile, the sum
        # lines are rewritten by _rewrite_promoted_rtree_body below (reshape-
        # collapse + all-r-slot resize). Suppress the generic single-slot sum
        # rewrites here so the two passes don't fight. tl.full / tl.broadcast_to
        # dense_size replacement still runs (shared with the x-path).
        _promoted_any = bool(getattr(kernel, "_npu_rtree_promoted", {}))

        for buf in [kernel.body, kernel.compute, kernel.stores,
                    kernel.post_loop_combine, kernel.post_loop_store]:
            for i, candidate in enumerate(buf._lines):
                if isinstance(candidate, str):
                    def _get():
                        return buf._lines[i]

                    def _set(new, _i=i):
                        buf._lines[_i] = new
                elif hasattr(candidate, "line") and isinstance(candidate.line, str):
                    def _get(_c=candidate):
                        return _c.line

                    def _set(new, _c=candidate):
                        _c.line = new
                else:
                    continue
                if needs_fixup:
                    if old_dense_size in _get():
                        _set(_get().replace(old_dense_size, real_dense_size))
                    if (not _promoted_any) and old_slice in _get() and f", {old_reduction_dim})" in _get():
                        _set(_get().replace(
                            f", {old_reduction_dim})", f", {real_reduction_dim})"
                        ).replace(old_slice, new_slice))
                    # Store-index broadcast fixup. Upstream store appends
                    # .broadcast_to(value.shape) with the pre-linearize shape (reduction
                    # slot=1, no brackets); after linearization the value is collapsed,
                    # so the stale target has the wrong rank. Rebuild old/new arg strings
                    # from old_sizes/real_sizes (reduction slot -> "1"), name/dtype-free.
                    old_val_sizes = list(old_sizes)
                    if 0 <= old_reduction_dim < len(old_val_sizes):
                        old_val_sizes[old_reduction_dim] = "1"
                    real_val_sizes = list(real_sizes)
                    if 0 <= real_reduction_dim < len(real_val_sizes):
                        real_val_sizes[real_reduction_dim] = "1"
                    old_bcast = f".broadcast_to({', '.join(old_val_sizes)})"
                    if old_val_sizes != real_val_sizes and old_bcast in _get():
                        new_bcast = f".broadcast_to({', '.join(real_val_sizes)})"
                        _set(_get().replace(old_bcast, new_bcast))
                    # The scalar store index is emitted as tl.full([1,...,1], v, dt)
                    # with one 1 per orig_ndim dim; after collapse it must carry
                    # real_ndim dims. Rewrite the rank prefix via orig_ndim/real_ndim,
                    # leaving value and dtype untouched.
                    old_scalar_rank = f"[{', '.join(['1'] * orig_ndim)}]"
                    real_scalar_rank = f"[{', '.join(['1'] * real_ndim)}]"
                    if (
                        orig_ndim != real_ndim
                        and f"tl.full({old_scalar_rank}, " in _get()
                    ):
                        _set(_get().replace(
                            f"tl.full({old_scalar_rank}, ",
                            f"tl.full({real_scalar_rank}, ",
                        ))
                # Independent dim fixup: upstream tl.sum(_, ndim-nreduce) hardcodes the
                # last slot. After the hook permutes R off it, the resize slice is right
                # but the dim arg inside tl.sum/max/min/triton_helpers still points at X.
                # Rewrite to R's slot, restricted to reduction-call lines so unrelated
                # ", 1)" (tl.full([1],...), tuples) aren't mangled.
                if (not _promoted_any) and old_reduction_dim != real_reduction_dim:
                    line = _get()
                    if (
                        f", {old_reduction_dim})" in line
                        and any(tok in line for tok in (
                            "tl.sum(", "tl.max(", "tl.min(", "tl.prod(",
                            "tl.argmax(", "tl.argmin(", "tl.reduce(",
                            "tl.xor_sum(", "triton_helpers.",
                        ))
                    ):
                        _set(line.replace(
                            f", {old_reduction_dim})",
                            f", {real_reduction_dim})",
                        ))
                if normalize_single_dim_reduction and "tl.full([XBLOCK, 1], 0, tl.int32)" in _get():
                    _set(_get().replace(
                        "tl.full([XBLOCK, 1], 0, tl.int32)",
                        "tl.full([1], 0, tl.int32)",
                    ))
                if 'tl.program_id(0)' in _get():
                    _set(_get().replace('tl.program_id(0)', '(group_base + i)'))

        # NOTE: broadcast_to stripping was previously done here for reduction
        # kernels, but it incorrectly removed broadcasts needed for correctness
        # (e.g. broadcasting mean [x0,1] to [x0,R0_BLOCK] for variance calc).
        # Degenerate scalar broadcasts (tl.broadcast_to(tmp, [1])) are handled
        # in codegen_kernel's final output instead.

        # Real-block multi-axis reduction rewrite (rtree_real_block). Per promoted
        # r-tree, turn the flat for-r0_offset loop + mod/div decomp into per-node
        # real_block tile aranges (full residency), and reshape-collapse the multi-slot
        # accumulator before each tl.sum. Runs AFTER the generic dense_size fixup; the
        # generic single-slot sum rewrite was suppressed above to avoid conflict.
        if getattr(kernel, "_npu_rtree_promoted", {}):
            # The flat ``for r0_offset`` loop does NOT exist in kernel.body yet
            # — it is generated later by codegen_body() from the pending
            # buffers. Stash the slot metadata so codegen_body can run the
            # real-block rewrite AFTER it materializes the loop.
            kernel._npu_rtree_rewrite_info = {
                "real_sizes": list(real_sizes),
                "real_ndim": real_ndim,
            }

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        """
        Override to track index_vars and var_ranges per node (needed for linearize mode).
        """
        if triton_codegen_linearize:
            kernel.var_ranges_per_node = []
            kernel.index_vars_per_node = []

        with kernel:
            stack = contextlib.ExitStack()
            all_indexing = {}

            # First pass: collect indexing and decide inplace updates
            for node in node_schedule:
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                else:
                    node.decide_inplace_update()
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    all_indexing.update(
                        dict.fromkeys(
                            node._body.indexing_from_args(index_vars).values()
                        )
                    )

            kernel.finalize_indexing(all_indexing.keys())

            # Second pass: actual codegen
            for node in node_schedule:
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                else:
                    from torch._inductor.optimize_indexing import indexing_dtype_strength_reduction
                    indexing_dtype_strength_reduction(node._body)
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())

                    if triton_codegen_linearize:
                        kernel.var_ranges_per_node.append(node.get_ranges())
                        kernel.index_vars_per_node.append(index_vars)

                    node.codegen(index_vars)


# =============================================================================
# Codegen / scheduler monkeypatches installed by overrides.apply_npu_overrides.
#
# These sit here (rather than in overrides.py) because they patch the Triton
# codegen / scheduler layer and share this module's imports (IterationRangesRoot,
# V, sympy, triton_codegen_linearize). overrides.apply_npu_overrides() calls
# apply_npu_codegen_patches() at backend activation.
# =============================================================================
def _npu_get_block_hint(tree: IterationRangesRoot):
    """Per-node block hints for an IterationRangesRoot: static lengths as ints, -1
    for dynamic, plus a tagged divisor payload so NPU autotune can synthesize
    high-stride-aligned XBLOCK candidates. E.g. [4, -1, -1, {"divisors": [1,4,16000]}]."""
    x = tree.prefix
    if x == 'r':
        raise AssertionError("Can not get_block_hint for rnumel!")
    block_hint = []
    divisor_hints = []
    for node in tree.nodes.values():
        if node.name in tree.tree_node_mapping:
            continue
        if isinstance(node.length, (int, sympy.Integer)):
            block_hint.append(int(node.length))
        else:
            block_hint.append(-1)

        if isinstance(node.divisor, (int, sympy.Integer)):
            divisor_val = int(node.divisor)
            divisor_hints.append(divisor_val)
        else:
            divisor_hint = -1
            try:
                divisor_hint = V.graph.sizevars.optimization_hint(node.divisor)
            except Exception:
                pass
            divisor_val = int(divisor_hint) if isinstance(divisor_hint, int) and divisor_hint > 0 else -1
            divisor_hints.append(divisor_val)

    block_hint.append({"divisors": divisor_hints})
    return block_hint


def _patch_zero_dim_cpu_tensor_for_npu():
    """Make NPU kernels unwrap 0-dim CPU tensor inputs to scalars.

    A 0-dim CPU scalar convert'd to f32 lowers to a host cpp kernel producing a
    0-dim CPU buffer; fed to an NPU triton kernel as a raw pointer it crashes
    ("Pointer argument cannot be accessed from Triton"). Inductor's remedy,
    Scheduler.update_zero_dim_cpu_tensor, gates on node.is_gpu() and GPU_TYPES
    excludes "npu", so it never fires. Rather than torch_npu's broad
    GPU_TYPES.append("npu"), narrowly re-implement it to also register 0-dim CPU
    buffers read by NPU nodes, leaving every other is_gpu path untouched.
    """
    from torch._inductor.scheduler import Scheduler
    from torch._inductor.ir import MultiOutputLayout, NoneLayout, get_device_type

    def _npu_update_zero_dim_cpu_tensor(self) -> None:
        for node in self.nodes:
            # Original gate is node.is_gpu(); also accept NPU nodes.
            dev = node.get_device()
            is_device_kernel = node.is_gpu() or (
                dev is not None and dev.type == "npu"
            )
            if not is_device_kernel:
                continue
            for read in node.read_writes.reads:
                buffer = V.graph.name_to_buffer.get(read.name)
                if (
                    buffer
                    and get_device_type(buffer) == "cpu"
                    and not isinstance(buffer.layout, (NoneLayout, MultiOutputLayout))
                    and buffer.get_size() == []
                ):
                    V.graph.zero_dim_cpu_tensor_list.add(read.name)

    Scheduler.update_zero_dim_cpu_tensor = _npu_update_zero_dim_cpu_tensor


def apply_npu_codegen_patches():
    """Install the NPU codegen/scheduler monkeypatches (called from overrides.py)."""
    if triton_codegen_linearize:
        # codegen_range_tree / iteration_ranges_* are real methods on
        # NPUTritonKernel; only attach get_block_hint here (used by NPUTritonScheduling).
        IterationRangesRoot.get_block_hint = _npu_get_block_hint

    _patch_zero_dim_cpu_tensor_for_npu()

    # NPU AI Vector Core does not natively support int64 arithmetic — demote to int32.
    import torch._inductor.utils as _inductor_utils
    _inductor_utils._triton_type_mapping["tl.int64"] = "tl.int32"
    # Also patch triton_compute_type which torch_npu overrides with its own
    # npu_triton_compute_type that bypasses _triton_type_mapping.
    import torch._inductor.codegen.triton as _triton_codegen
    _orig_compute_type = _triton_codegen.triton_compute_type

    def _npu_triton_compute_type(dtype):
        result = _orig_compute_type(dtype)
        if result == "tl.int64":
            return "tl.int32"
        return result

    _triton_codegen.triton_compute_type = _npu_triton_compute_type

    import torch._inductor.runtime.triton_heuristics as _triton_heuristics

    def _npu_check_config(cfg, *, xnumel=None, ynumel=None, znumel=None):
        for numel, label in zip((xnumel, ynumel, znumel), "XYZ"):
            if numel is None:
                continue
            block = cfg[f"{label}BLOCK"]
            if numel == 1 and block != 1:
                raise RuntimeError(
                    "[triton_experimental] TritonKernel.indexing assumes numel == 1 => BLOCK == 1"
                    f" but {label.lower()}numel=={numel} and {label}BLOCK={block} (cfg={cfg})."
                )

    _triton_heuristics.check_config = _npu_check_config
