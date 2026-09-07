# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Permute-gather reduction rewrite (ncfg.enable_permute_gather), extracted from
# codegen/triton.py per review: all rewrite logic lives here (sections:
# eligibility/geometry/validation, then the gather and trans emissions);
# NPUTritonKernel keeps thin delegates so call sites and test monkeypatching
# are unchanged.

import ast
import logging
import operator
import sympy

from torch._inductor.virtualized import V

from .. import config as ncfg
from .. import device_props

log = logging.getLogger(__name__)


def _ast_helpers():
    # Deferred: .triton imports this module lazily (from its delegates), so a
    # module-level back-import of its line-parsing helpers would be circular.
    global _parse_line, _assignment_parts, _parse_tl_load
    from torch_npu._inductor.triton_experimental.codegen.triton import (
        _npu_parse_generated_line as _parse_line,
        _npu_assignment_parts as _assignment_parts,
        _npu_parse_tl_load_assignment as _parse_tl_load,
    )


# Permute-gather rewrite UB budget (_npu_pg_geometry). The rewrite holds three
# co-resident tiles of stride_r*R0_BLOCK elements -- flat (input dtype), the
# int32 gather index, and the gathered result -- and triton caps tensor numel at
# 2^20. When the full row exceeds either limit the reduction is tiled into
# multiple r-loop trips (per-chunk load, see _npu_pg_emit). The largest R0_BLOCK
# that fits both caps is pinned into triton_meta (the rewrite forbids an autotune
# sweep), so these constants tune the tile at its widest:
#   * ELEM_BYTES: worst-case fp32 for all three tiles (int32 index is always 4B);
#     fp16/fp8 inputs get a conservative (smaller) tile, just more trips.
#   * PIPE: the compiler doubles live buffers for software pipelining (matches
#     _NPU_UB_OVERHEAD_FACTOR in npu_triton_heuristics).
#   * RESERVE: keep a share of UB for the rest of the kernel (other loads, output,
#     masks). Under-estimating only costs extra trips -- it never breaks compile.
_PG_UB_RESERVE = 0.5
_PG_UB_TENSORS = 3
_PG_UB_ELEM_BYTES = 4
_PG_UB_PIPE = 2

# _npu_pg_eval_expr operator tables (lookup == the if-chain it replaces)
_PG_EVAL_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_PG_EVAL_CMPOPS = {
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _npu_pg_eval_expr(node, env):
    """Evaluate a pre_loop constexpr RHS (int arithmetic over ``env`` names).

    A tiny AST walker instead of eval(): the emitted tile-chain lines contain
    only integer literals, bound names, + - * // %, comparisons, the
    ``a if cond else b`` tile form and min/max; anything else raises and the
    caller skips the line.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]  # KeyError -> caller skips this line
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_npu_pg_eval_expr(node.operand, env)
    if isinstance(node, ast.BinOp) and type(node.op) in _PG_EVAL_BINOPS:
        fn = _PG_EVAL_BINOPS[type(node.op)]
        return fn(_npu_pg_eval_expr(node.left, env), _npu_pg_eval_expr(node.right, env))
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and type(node.ops[0]) in _PG_EVAL_CMPOPS
    ):
        fn = _PG_EVAL_CMPOPS[type(node.ops[0])]
        return fn(
            _npu_pg_eval_expr(node.left, env),
            _npu_pg_eval_expr(node.comparators[0], env),
        )
    if isinstance(node, ast.IfExp):
        branch = node.body if _npu_pg_eval_expr(node.test, env) else node.orelse
        return _npu_pg_eval_expr(branch, env)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("min", "max")
        and not node.keywords
    ):
        vals = [_npu_pg_eval_expr(arg, env) for arg in node.args]
        return min(vals) if node.func.id == "min" else max(vals)
    raise ValueError(f"unsupported constexpr expr: {ast.unparse(node)}")


# ---- eligibility / geometry / tiling validation (the dispatch half) --------


def _npu_pg_record(kernel, result_var, name, index):
    """Register an eligible strided reduction load for the gather rewrite."""
    if not ncfg.enable_permute_gather:
        return
    if not ncfg.codegen_linearize:
        return
    if not kernel.inside_reduction:
        return
    if getattr(kernel, "persistent_reduction", False):
        return
    var_name = getattr(result_var, "name", None)
    if not var_name:
        return
    try:
        # Use the buffer dtype (native itemsize). result_var.dtype is the
        # fp32-promoted codegen dtype (codegen_upcast_to_fp32), which would
        # dispatch fp16/bf16 permutes at 4B/elt -- same mode only for H%16==0,
        # and a wrong mode elsewhere (lowering/npu_permute dispatches on the
        # native 2B itemsize, _elemsize_of).
        elemsize = V.graph.get_dtype(name).itemsize
    except Exception:
        elemsize = 4
    geo = _npu_pg_geometry(kernel, index, elemsize)
    if geo is None:
        log.debug("permute_gather: %s ineligible (geometry/mode gate)", name)
        return
    if not hasattr(kernel, "_npu_pg_candidates"):
        kernel._npu_pg_candidates = {}
    kernel._npu_pg_candidates[var_name] = geo
    log.debug("permute_gather: candidate %s mode=%s stride_r=%s",
              name, geo["mode"], geo["stride_r"])


def _npu_pg_geometry(kernel, index, elemsize=4):
    """Return the rewrite geometry for ``index``, or None if not eligible.

    Single-interior geometry: exactly one reduction leaf with coeff
    stride_r > 1, and exactly two free leaves -- interior (coeff 1) and row
    (coeff == SEG = stride_r * r_numel) -- in one x-tree, with the interior
    node running exactly stride_r elements. Anything else falls back to the
    strided load (correct, just slow). Mode semantics ("gather" = flat DMA +
    register tl.gather vs "trans" = block_ptr + tl.trans riding the MTE2
    read, per-branch caps, gather's static-stride_r requirement): see
    config.permute_gather_mode and the _npu_pg_emit* docstrings.
    """
    leaves = {}
    size_syms = set()
    red_sym = None
    red_leaf = None
    for sym in index.free_symbols:
        node = kernel.range_tree_nodes.get(sym)
        if node is None:
            # A symbolic size parameter (dynamic dim), e.g. dynamic Sk riding
            # the row-leaf coeff stride_r * r_numel. Validated against the
            # symbolic r_numel below; must not appear anywhere else (the
            # residue check rejects a dynamic base offset). Deferred instead
            # of rejected so dynamic reduction numels reach the row-leaf
            # compare and can take the runtime-mask gather path.
            size_syms.add(sym)
            continue
        try:
            coeff = index.coeff(sym, 1)
        except Exception:
            return None
        if node.root.is_reduction:
            if red_leaf is not None:
                return None  # >1 reduction leaf -> not single-interior
            if not isinstance(coeff, (int, sympy.Integer)):
                # Dynamic H: the reduction stride is symbolic (e.g. ks0 in
                # x1 + ks0*r0_3 + ks0*ks1*x0). The gather index must be
                # compile-time affine in stride_r (vectorized UB permute) and
                # the flat tile width stride_r*R0_BLOCK constexpr, so a
                # symbolic stride can only ride the trans mode's shape-generic
                # block_ptr (mode dispatch below).
                if not ncfg.permute_gather_dynamic_trans:
                    return None
                red_leaf = (node, coeff)
                red_sym = sym
                continue
            red_leaf = (node, int(coeff))
            red_sym = sym
        else:
            # free-leaf coeff may be symbolic (e.g. stride_r * dynamic r_numel
            # on the row leaf); validated against stride_r * r_numel_expr below
            leaves[sym] = (node, coeff)
    if red_leaf is None:
        return None
    rnode, stride_r = red_leaf
    if isinstance(stride_r, sympy.Expr) and stride_r.free_symbols:
        dynamic_h = True
        mode = "trans"
    else:
        stride_r = int(stride_r)
        dynamic_h = False
        if stride_r <= 1:
            return None  # unit-stride reduction is already contiguous
        # Mode dispatch (mirrors lowering.npu_permute): inner strides below
        # 256B (H<=63 fp32) -> register gather; >= 256B (H>=64 fp32) ->
        # block_ptr trans, which beats both the old gather (flat tile
        # overflows UB, measured H=64 R0=128 compile fail) and the realized
        # transpose (two-kernel 219us) on the MTE2 permutation.
        mode = ncfg.permute_gather_mode(stride_r, elemsize)
        if mode is None:
            # stride_r above the XBLOCK cap (4096): neither rewrite can pin
            # XBLOCK = stride_r, so keep the strided load -- lowering has
            # realized the permute (mode None -> realize), making it correct.
            return None
    r_len = rnode.length
    if isinstance(r_len, (int, sympy.Integer)):
        r_numel = int(r_len)
        if r_numel <= 0:
            return None
        r_numel_expr = sympy.Integer(r_numel)
        dynamic_r = False
    else:
        # Dynamic reduction numel: the runtime masks (flat seg bound in
        # _npu_pg_emit + the reduction's own r0_mask) keep every output lane
        # exact for any r_numel, so the rewrite stays sound. r0_block is then
        # sized from the UB caps alone (can't min against a runtime numel);
        # since r0_mask is runtime (never the all-True constant-mask fold),
        # any pow2 R0_BLOCK is safe -- no divisibility requirement.
        if r_len.free_symbols - size_syms:
            return None
        r_numel = None
        r_numel_expr = r_len
        dynamic_r = True
    # Affine residue: removing every linear term must leave a pure constant
    # base offset (preserved verbatim in the emit); any remaining free
    # symbol would be a dynamic base offset the flat load cannot reproduce.
    linear = sum(c * s for s, (n, c) in leaves.items())
    # stride_r is a Python int (static H) or a sympy Symbol (dynamic H); the
    # plain multiply covers both, avoiding sympy.Integer(Symbol) (TypeError).
    linear += stride_r * red_sym
    residue = index - linear
    if not isinstance(residue, sympy.Number):
        return None
    if len(leaves) != 2:
        return None
    int_leaf = row_leaf = None
    for s, (node, coeff) in leaves.items():
        if coeff == 1:
            int_leaf = (node, coeff)
        elif coeff == stride_r * r_numel_expr:
            row_leaf = (node, coeff)
        else:
            return None  # extra / unexpected free stride
    if int_leaf is None or row_leaf is None:
        return None
    int_node, _ = int_leaf
    row_node, _ = row_leaf
    if row_node.root is not int_node.root:
        return None  # interior and row must share one x-tree
    if dynamic_h:
        # Dynamic H: interior length is the symbolic head axis (== stride_r)
        if int_node.length != stride_r:
            return None
    elif not isinstance(int_node.length, (int, sympy.Integer)) or int(int_node.length) != stride_r:
        return None
    r_tree = rnode.root
    # Register-dim slots (var_tensor_dims) are assigned by _apply_linearize,
    # which runs AFTER the first body pass emits loads -- so slot resolution is
    # deferred to _npu_pg_rewrite_body (codegen_kernel time), when they exist.
    #
    # Reduction tile for the rewrite: largest R0_BLOCK that keeps the emitted
    # per-chunk flat/idx/gather tiles (width stride_r * R0_BLOCK, see
    # _npu_pg_emit) inside triton's 2^20 tensor-numel cap and UB. Larger tile =
    # fewer r-loop trips, so take the max. At R0_BLOCK == r_numel the loop
    # runs once and the emission degenerates to the original whole-row burst.
    #
    # R0_BLOCK must be a power of two.  Upstream's constant-mask optimization
    # (_has_constant_mask) emits r0_mask = tl.full(..., True) whenever the max
    # reduction block (TRITON_MAX_BLOCK["R0_"] = 65536) divides r_numel, and
    # that is only sound if the ACTUAL R0_BLOCK also divides r_numel -- true
    # for upstream's pow2 blocks, broken by a non-pow2 tile (e.g. 341) which
    # leaves a partial tail trip whose all-True mask accumulates out-of-bounds
    # lanes into the reduction.  Round the raw cap DOWN to the largest pow2
    # (cap_ub <= 4096//stride_r <= 2048, so the tile also divides 65536 and
    # hence every multiple of it).
    if dynamic_h:
        # Dynamic-H trans is config-agnostic (block_shape adapts to the swept
        # (XBLOCK, R0_BLOCK) and boundary_check keeps partial chunks exact),
        # so no config pin and no r0_block.
        r0_block = None
    else:
        # Effective interior width resident in the tile. Trans chunks the int
        # axis to permute_gather_ktile (the XBLOCK pin), so the caps must run
        # on the chunk width -- this is what decouples R0 from H (H=4096:
        # R0 1 -> 64). Gather's flat tile is stride_r wide by construction
        # (XBLOCK pin = stride_r, no chunking), so eff == stride_r.
        eff = stride_r if mode == "gather" else min(stride_r, ncfg.permute_gather_ktile)
        cap_numel = 1_048_576 // eff
        budget = int(_PG_UB_RESERVE * device_props.get_npu_ub_size_bytes())
        cap_ub = max(1, budget // (_PG_UB_TENSORS * _PG_UB_ELEM_BYTES * eff * _PG_UB_PIPE))
        raw = min(cap_numel, cap_ub) if r_numel is None else min(r_numel, cap_numel, cap_ub)
        r0_block = 1 << (max(1, raw).bit_length() - 1)
    return {
        "mode": mode,
        "dynamic_h": dynamic_h,
        "int_node": int_node,
        "row_node": row_node,
        "r_tree": r_tree,
        "rprefix": r_tree.prefix,
        "rblk": f"{r_tree.prefix.upper()}BLOCK",
        "stride_r": stride_r,
        "r_numel": r_numel,
        "seg": stride_r * r_numel if r_numel is not None else None,
        "dynamic_r": dynamic_r,
        "const": int(residue),
        "r0_block": r0_block,
    }


def _npu_pg_eval_real_blocks(tree, xblock):
    """Evaluate the emitted pre_loop constexprs under a forced XBLOCK.

    Returns {name: value} over the tile chain (real_block_*, *_numel,
    *_blocks) by AST-parsing the pre_loop assignment lines (AnnAssign
    ``name : tl.constexpr = ...`` normalized like _npu_build_grid_recipe)
    and walking each RHS with _npu_pg_eval_expr over the names bound so
    far. Directly reflects the runtime values, immune to ordering-logic
    drift; a first definition wins, as before.
    """
    _ast_helpers()
    env = {"XBLOCK": int(xblock)}
    pre = getattr(tree, "pre_loop_code", None)
    if pre is None:
        return env
    for entry in pre._lines:
        line = entry if isinstance(entry, str) else getattr(entry, "line", None)
        if not isinstance(line, str):
            continue
        parsed = _parse_line(line)
        if parsed is None:
            continue
        _, statement = parsed
        if isinstance(statement, ast.AnnAssign) and statement.value is not None:
            statement = ast.Assign(
                targets=[statement.target], value=statement.value
            )
        assignment = _assignment_parts(statement)
        if assignment is None or assignment[0] in env:
            continue
        try:
            env[assignment[0]] = _npu_pg_eval_expr(assignment[1], env)
        except Exception:
            continue
    return env


def _npu_pg_rewrite_body(kernel):
    """Replace the recorded strided reduction load with DMA + tl.gather, or
    block_ptr + tl.trans (post codegen_body). Validates the forced tiling
    against the emitted pre_loop constexprs (gather: XBLOCK=stride_r -> row
    tile 1 + full int run; trans: XBLOCK=min(stride_r, ktile) -> row tile 1,
    int chunked), then swaps the load line. Returns the forced config dict
    for triton_meta, or None to keep strided / leave the sweep free (dyn-H).
    """
    _ast_helpers()
    cands = getattr(kernel, "_npu_pg_candidates", None) or {}
    if len(cands) != 1:
        return None  # the per-kernel config pin needs exactly one candidate
    var_name, geo0 = next(iter(cands.items()))
    # Register-dim slots exist only after _apply_linearize (this hook runs at
    # codegen_kernel time, so they're present now).
    vtd = geo0["int_node"].root.var_tensor_dims
    r_tree = geo0["r_tree"]
    int_slot = vtd.get(geo0["int_node"].name)
    row_slot = vtd.get(geo0["row_node"].name)
    r_slot = getattr(r_tree, "tensor_dim", None)
    # Slots must be distinct and in range; no relative-order assumption.
    # The emit derives everything from the slots themselves (trans perm via
    # sorted slot order, reshape shape filled per-slot), so it stays correct
    # for any layout -- including the reduction tree taking slot 0, which the
    # default-backend scheduler patches produce for large H kernels
    # (range_trees R-first + _npu_repermute_tensor_dims stride order). The
    # historical int/row-before-r check came from the hard-coded (0,2,1)
    # trans perm and silently dropped those kernels back to strided loads.
    slots = (int_slot, row_slot, r_slot)
    # Type-check before max(): a legitimately-missing slot (vtd key
    # absent / r_tree.tensor_dim None) must fall back to strided, not
    # crash codegen with a TypeError on max((None, ...)).
    if not all(isinstance(s, int) for s in slots):
        log.debug("permute_gather: %s slot missing -> strided", var_name)
        return None
    ndim = max(slots) + 1
    if not all(0 <= s < ndim for s in slots) or len(set(slots)) != 3:
        log.debug("permute_gather: %s slots not distinct/in-range -> strided", var_name)
        return None
    geo = dict(geo0)
    geo["int_slot"] = int_slot
    geo["row_slot"] = row_slot
    geo["r_slot"] = r_slot
    geo["ndim"] = ndim
    tree = geo0["int_node"].root
    # Static modes pin XBLOCK so the greedy tile chain produces exactly the
    # tiling the rewrite needs; the rewrite is only valid under that tiling.
    # Gather pins XBLOCK=stride_r: the flat tile + pidx reshape structurally
    # need real_block_int == stride_r == int_numel. Trans pins
    # XBLOCK=min(stride_r, permute_gather_ktile): the interior axis is chunked
    # across programs (x1_blocks > 1) and the block_ptr/reshape are
    # chunk-agnostic (real_block_int in both), but the row axis must stay
    # tile-1 (the reshape's [1, rb_int, rb_row, R] broadcast would break) --
    # the int axis eating the whole pin budget guarantees that. Dynamic-H
    # trans is config-agnostic (the block_ptr block_shape adapts and
    # boundary_check guards partial chunks), so no pin and no tiling invariant.
    if not geo0["dynamic_h"]:
        if geo0["mode"] == "gather":
            pin = geo0["stride_r"]
        else:
            pin = min(geo0["stride_r"], ncfg.permute_gather_ktile)
        env = _npu_pg_eval_real_blocks(tree, pin)
        rb_row = env.get(f"real_block_{geo0['row_node'].name}")
        rb_int = env.get(f"real_block_{geo0['int_node'].name}")
        int_numel = env.get(f"{geo0['int_node'].name}numel")
        if rb_row != 1 or rb_int != min(int_numel, pin):
            log.debug("permute_gather: %s tiling mismatch (rb_row=%s rb_int=%s, "
                      "want row=1 int<=%s) -> strided", var_name, rb_row, rb_int, pin)
            return None  # tiling not as forced -> keep strided (correct, slow)
    # Locate the recorded load structurally (same pattern as
    # _maybe_rewrite_select_lane_load): match the assignment TARGET, not the
    # line text -- a load line may carry a trailing ``.to(tl.float32)``
    # promotion that no ``tl.load(...)$`` regex can delimit, and non-str
    # body entries (DeferredLine et al.) must pass through untouched.
    new_lines = []
    rewritten = False
    for line in kernel.body._lines:
        if not rewritten and isinstance(line, str):
            parsed = _parse_tl_load(line)
            if parsed is not None and parsed[1] == var_name:
                indent, _, value_ast, load_ast = parsed
                emit = _npu_pg_emit(
                    var_name, indent, value_ast, load_ast, geo, mode=geo0["mode"]
                )
                if emit:
                    new_lines.extend(emit)
                    rewritten = True
                    continue
                # unexpected load shape -> keep the strided line below
        new_lines.append(line)
    if not rewritten:
        log.debug("permute_gather: %s load line not found -> strided", var_name)
        return None
    kernel.body._lines = new_lines
    log.debug("permute_gather: rewrote %s (%s)", var_name, geo0["mode"])
    if geo0["dynamic_h"]:
        return None  # dynamic-H trans: no config pin, autotune sweeps freely
    if geo0["mode"] == "gather":
        return {"XBLOCK": geo0["stride_r"], "R0_BLOCK": geo0["r0_block"]}
    return {
        "XBLOCK": min(geo0["stride_r"], ncfg.permute_gather_ktile),
        "R0_BLOCK": geo0["r0_block"],
    }


# ---- emission: "gather" mode (flat contiguous DMA + register tl.gather) ----


def _npu_pg_emit(var_name, indent, value_ast, load_ast, geo, mode="gather"):
    """Build replacement lines for one permute rewrite load.

    ``value_ast``/``load_ast`` come from _npu_parse_tl_load_assignment on the
    emitted line: the full assignment value (possibly a ``.to(tl.float32)``
    promotion wrapped around the call) and the ``tl.load`` call node itself.
    The base pointer, ``other`` and ``eviction_policy`` are read structurally
    off the call node -- never by regex over the rendered text (an ``other``
    value containing parens, e.g. ``float('-inf')``, defeats any ``[^)]*``
    capture). Returns [] to decline the rewrite (keep the strided line).

    ``mode`` selects the primitive: "gather" (flat DMA + register tl.gather)
    or "trans" (block_ptr [row, r, int] + tl.trans, see _npu_pg_emit_trans).
    """
    # args[0] is the emitted ``ptr + (linear index)``; the rewrite rebuilds
    # the address from the geometry, so only the bare pointer name is needed.
    if not (
        isinstance(load_ast.args[0], ast.BinOp)
        and isinstance(load_ast.args[0].op, ast.Add)
        and isinstance(load_ast.args[0].left, ast.Name)
    ):
        return []
    ptr = load_ast.args[0].left.id
    other = "0.0"
    evict = "evict_last"
    for kw in load_ast.keywords:
        if kw.arg == "other":
            other = ast.unparse(kw.value)
        elif kw.arg == "eviction_policy":
            try:
                evict = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                return []
    if mode == "trans":
        return _npu_pg_emit_trans(
            var_name, indent, ptr, geo, other, value_ast, load_ast
        )
    int_n = geo["int_node"].name
    row_n = geo["row_node"].name
    rpfx = geo["rprefix"]
    rblk = geo["rblk"]
    s = geo["stride_r"]
    # Row-bound of the source buffer (seg = stride_r * r_numel). Static
    # r_numel: bake the Python int. Dynamic r_numel: the runtime value
    # stride_r * r0_numel (the very numel the r-loop and r0_mask iterate),
    # so the flat mask + gather stay exact for any sequence length -- the
    # mask-based precision guarantee that lets dynamic Sk ride this path.
    seg = f"{s} * {rpfx}numel" if geo.get("dynamic_r") else str(geo["seg"])
    base = f"{ptr} + {geo['const']} + " if geo["const"] else f"{ptr} + "
    rows = f"{var_name}_pg_rows"
    flat = f"{var_name}_pg_flat"
    pr = f"{var_name}_pg_r"
    pidx = f"{var_name}_pg_idx"
    pg = f"{var_name}_pg_g"
    # The var's reshape target must be r-last: the surrounding reduction
    # code consumes it by axis POSITION, not by register slot -- r0_mask is
    # [1, 1, 1, R0_BLOCK], the broadcast_to / tl.sum run over the trailing
    # r axis. Filling ``parts`` per-slot was only right for the int<row<r
    # layout by coincidence; R-first layouts (r_slot < int_slot) then put
    # r at slot 0 and either broke the broadcast or silently transposed the
    # permuted values. Fix the shape as [1, int, row, R0_BLOCK] and build
    # pidx in the matching [int, r] flat order -- independent of the slots.
    shape = f"[1, real_block_{int_n}, real_block_{row_n}, {rblk}]"
    # Per-chunk DMA: this trip of the r-loop loads only its own
    # stride_r*R0_BLOCK stretch of the row, at r0_offset*stride_r past the
    # row base. At R0_BLOCK == r_numel (single trip, r0_offset == 0) this is
    # exactly the original whole-row burst. The flat r-bound mask keeps the
    # tail trip's load lanes inside the row (never reads past seg); the
    # reduction's own r0_mask still zeroes the corresponding outputs.
    tile = f"{rpfx}offset * {s} + tl.arange(0, {s} * {rblk})"
    lines = [
        f"{indent}{rows} = {row_n}offset + tl.arange(0, real_block_{row_n})",
        f"{indent}{flat} = tl.load({base}{seg} * {rows}[:, None] + {tile}[None, :], "
        f"({rows}[:, None] < {row_n}numel) & ({tile} < {seg}), "
        f"eviction_policy='{evict}', other={other})",
        f"{indent}{pr} = tl.arange(0, {rblk})[None, :]",
    ]
    # pidx's grid is [int, r] (row-major flat = int outer, r inner), which
    # matches the r-last reshape's flatten exactly. No runtime guard is
    # emitted: the rewrite only produces this code after its own tiling
    # check (real_block_row == 1, real_block_int == stride_r) passed at
    # rewrite time, and Python `raise` is invalid Triton kernel code.
    lines.append(
        f"{indent}{pidx} = tl.reshape({int_n}offset + tl.arange(0, real_block_{int_n})[:, None] "
        f"+ {s} * {pr}, (1, {s} * {rblk})) + tl.full([real_block_{row_n}, 1], 0, tl.int32)"
    )
    lines.append(f"{indent}{pg} = tl.reshape(tl.gather({flat}, {pidx}, 1), {shape})")
    lines.append(
        _npu_pg_final_line(
            var_name, indent, value_ast, load_ast, f"tl.where({rpfx}mask, {pg}, {other})"
        )
    )
    return lines


def _npu_pg_final_line(var_name, indent, value_ast, load_ast, expr_text):
    """``var = expr_text`` with the load's surrounding expression preserved:
    splice the replacement in place of the ``tl.load`` node and unparse, so
    an fp16/bf16 candidate keeps its ``.to(tl.float32)`` promotion (with no
    promotion this unparses to exactly ``expr_text``)."""
    repl = ast.parse(expr_text, mode="eval").body

    class _SwapLoad(ast.NodeTransformer):
        def visit_Call(self, node):
            if node is load_ast:
                return ast.copy_location(repl, node)
            return self.generic_visit(node)

    value_ast = _SwapLoad().visit(value_ast)
    ast.fix_missing_locations(value_ast)
    return f"{indent}{var_name} = {ast.unparse(value_ast)}"


# ---- emission: "trans" mode (block_ptr + tl.trans riding the MTE2 read) ----


def _npu_pg_emit_trans(var_name, indent, ptr, geo, other, value_ast, load_ast):
    """block_ptr [row, r, int] + tl.trans + reshape, replacing the strided
    reduction load. Shape/strides ride the always-present per-axis numel vars
    (stride_r == int_numel and seg == int_numel * r_numel by the geometry's
    leaf invariants), so one emission covers static and dynamic shapes;
    boundary_check + padding_option="zero" keeps tail chunks exact for any
    constexpr block. The permutation rides the MTE2 read (a UB tile permute)
    -- no index tensor, so lower UB pressure at large stride_r than gather.
    """
    int_n = geo["int_node"].name
    row_n = geo["row_node"].name
    rpfx = geo["rprefix"]
    rblk = geo["rblk"]

    def _numel(node):
        # Per-node numel args are emitted only for dynamic nodes (triton.py
        # size-arg loop); bake a static length as the literal int.
        if isinstance(node.length, (int, sympy.Integer)):
            return str(int(node.length))
        return f"{node.name}numel"

    int_nm = _numel(geo["int_node"])
    row_nm = _numel(geo["row_node"])
    base = f"{ptr} + {geo['const']}" if geo["const"] else ptr
    bp = f"{var_name}_pg_bp"
    t = f"{var_name}_pg_t"
    g = f"{var_name}_pg_g"
    parts = ["1"] * geo["ndim"]
    parts[geo["int_slot"]] = f"real_block_{int_n}"
    parts[geo["row_slot"]] = f"real_block_{row_n}"
    parts[geo["r_slot"]] = rblk
    shape = "[" + ", ".join(parts) + "]"
    # The loaded tile t is physical (row, r, int); the reshape target is
    # slot-ascending, so the trans perm must lay the axes in slot order
    # (a hard-coded (0,2,1) scrambles R-first layouts: measured dynH
    # diff=109). Derive the perm from the actual slots.
    pos = {"row": 0, "r": 1, "int": 2}
    perm = tuple(pos[k] for k in sorted(pos, key=lambda k: geo[f"{k}_slot"]))
    return [
        f"{indent}{bp} = tl.make_block_ptr(",
        f"{indent}    {base}, shape=[{row_nm}, {rpfx}numel, {int_nm}], "
        f"strides=[{int_nm} * {rpfx}numel, {int_nm}, 1], "
        f"offsets=[{row_n}offset, {rpfx}offset, {int_n}offset], "
        f"block_shape=[real_block_{row_n}, {rblk}, real_block_{int_n}], order=[2, 1, 0])",
        f'{indent}{t} = tl.load({bp}, boundary_check=[0, 1, 2], padding_option="zero")',
        f"{indent}{g} = tl.reshape(tl.trans({t}, {perm}), {shape})",
        _npu_pg_final_line(
            var_name, indent, value_ast, load_ast, f"tl.where({rpfx}mask, {g}, {other})"
        ),
    ]
