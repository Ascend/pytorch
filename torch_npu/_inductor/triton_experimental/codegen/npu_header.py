# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""NPU linearize-mode Triton header generation for triton_experimental.

Extracted from ``npu_patch.py``: this module is *not* a monkeypatch. It owns the
per-``IterationRangesRoot`` header code that ``codegen/triton.py`` emits when the
NPU backend runs in linearize mode (``config.codegen_linearize``). Kept beside its
sole caller (``codegen/triton.py``) to avoid a load-time import cycle.
"""

from .. import config as ncfg

import sympy

from torch._inductor.virtualized import V
from torch._inductor.utils import IndentedBuffer
from torch._inductor.codegen.triton import texpr

# Vector CMP on A2/A3 lacks native int64/int32, so an int arange < numel decays to
# a scalar loop; casting the index to fp32 first keeps it on the vector unit.
# Default off: fp32's 24-bit mantissa loses precision past 2^24, so only opt in
# (mask_cmp_fp32) when all per-axis numels stay below ~2^23.
npu_mask_cmp_fp32 = ncfg.mask_cmp_fp32


def _mask_cmp_lhs(index_expr: str) -> str:
    """Wrap a mask LHS so the `<` runs on the vector unit (fp32) when possible."""
    if not npu_mask_cmp_fp32:
        return index_expr
    return f"({index_expr}).to(tl.float32)"


def _ordered_mapping_items(mapping):
    """Yield (name, expr) entries in dependency order: any name referenced by
    another mapping's expression is emitted before that mapping. The dict's
    insertion order isn't reliable — fused-axis splits and flattened-view
    mappings can register entries whose expressions reference each other."""
    remaining = dict(mapping)
    emitted = set()
    while remaining:
        progressed = False
        for name in list(remaining):
            expr = remaining[name]
            deps = {str(s) for s in expr.free_symbols} & set(remaining)
            deps.discard(name)
            if not deps:
                yield name, expr
                emitted.add(name)
                del remaining[name]
                progressed = True
        if not progressed:
            for name, expr in remaining.items():
                yield name, expr
            return


def _render_mapping_expr(kernel, tree, tree_expr):
    """Render a tree_node_mapping expression for codegen. Returns
    (expr_string, iter_var_names). _FlatMapExpr is pre-rendered; a raw sympy expr
    has its size symbols translated to kernel-arg names via rename_indexing."""
    from .triton import _FlatMapExpr
    if isinstance(tree_expr, _FlatMapExpr):
        iter_vars = sorted(str(s) for s in tree_expr.free_symbols)
        return str(tree_expr), iter_vars

    renamed = kernel.rename_indexing(tree_expr)
    expr_str = texpr(renamed)
    # Only iteration variable symbols contribute to the mask.
    all_node_names = {n.name for n in tree.nodes.values()}
    iter_vars = sorted(
        str(s) for s in tree_expr.free_symbols if str(s) in all_node_names
    )
    return expr_str, iter_vars


def _npu_coeff_rank(coeff_expr, node_syms):
    """Rank a load-address coefficient by memory stride: a sortable
    ``(degree, const)`` where degree is the polynomial degree in the dynamic size
    symbols and const the numeric multiplier. Size symbols are large positive
    ints, so higher degree => larger stride (1 < ks0 < 16*ks0 < 16*ks0**2), i.e.
    most-contiguous first. Returns None for a zero coeff or a cross term (another
    iter var present)."""
    if coeff_expr == 0:
        return None
    # A clean stride is linear in this one node var only; another node var => cross.
    if coeff_expr.free_symbols & node_syms:
        return None
    try:
        const, rest = coeff_expr.as_coeff_Mul()
    except Exception:
        return None
    fs = rest.free_symbols
    if not fs:
        deg = 0
    else:
        try:
            deg = sum(int(sympy.degree(rest, s)) for s in fs)
        except Exception:
            deg = 1 << 20  # non-polynomial (floor/mod): rank last
    try:
        cnum = abs(float(const))
    except Exception:
        cnum = float("inf")
    if cnum == 0:
        return None
    return (deg, cnum)


def _npu_extract_load_addr(line):
    """Extract the address sub-expression from a ``tl.load(...)`` line as a string
    (for sympify), or None if unparseable. Bracket-balanced, not regex: the addr
    may contain commas (``div_floor_integer(x0, ks1)``) that a non-greedy regex
    would truncate, so scan paren depth for the top-level comma / '+' instead."""
    key = "tl.load("
    start = line.find(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 0
    arg_end = -1
    # Walk to the first top-level comma (mask separator) or the matching ')'.
    while i < len(line):
        c = line[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:        # matching close of tl.load( with no mask
                arg_end = i
                break
            depth -= 1
        elif c == "," and depth == 0:
            arg_end = i
            break
        i += 1
    if arg_end < 0:
        return None
    arg = line[start + len(key):arg_end].strip()
    # arg == "ptr + (addr)" / "ptr + addr": split the pointer at the top-level '+'.
    depth = 0
    plus = -1
    for j, c in enumerate(arg):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "+" and depth == 0:
            plus = j
            break
    if plus < 0:
        return None
    addr = arg[plus + 1:].strip()
    # Strip one fully-enclosing paren pair: "(expr)" -> "expr".
    if addr.startswith("(") and addr.endswith(")"):
        depth = 0
        wraps = True
        for k, c in enumerate(addr):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and k != len(addr) - 1:
                    wraps = False
                    break
        if wraps:
            addr = addr[1:-1].strip()
    return addr or None


def _npu_harvest_node_input_strides(kernel, free_tile_nodes):
    """Per-node input memory-stride RANK, parsed from emitted ``tl.load`` addrs.
    Returns {node_name: rank} using _npu_coeff_rank on each node's iter var; when a
    var appears in several loads, keep the smallest-stride (most-contiguous) rank.
    Reads kernel.loads; returns {} on any failure (caller falls back to divisor)."""
    buf = getattr(kernel, "loads", None)
    lines = getattr(buf, "_lines", None) if buf is not None else None
    if not lines:
        return {}
    names = [n.name for n in free_tile_nodes]
    syms = {nm: sympy.Symbol(nm) for nm in names}
    node_syms = set(syms.values())
    result = {}
    for ln in lines:
        s = ln if isinstance(ln, str) else getattr(ln, "line", "")
        if "tl.load" not in s:
            continue
        addr = _npu_extract_load_addr(s)
        if addr is None:
            continue
        try:
            expr = sympy.sympify(addr, locals=syms)
        except Exception:
            continue
        for nm, sym in syms.items():
            try:
                coeff = expr.coeff(sym)
            except Exception:
                continue
            rank = _npu_coeff_rank(coeff, node_syms)
            if rank is None:
                continue
            prev = result.get(nm)
            if prev is None or rank < prev:
                result[nm] = rank
    return result


def _codegen_header_npu_for_tree(kernel, tree, code, outer_blocks=None):
    """NPU linearize-mode header generator for a single IterationRangesRoot
    (kernel and tree passed explicitly)."""
    header_code = IndentedBuffer()
    x = tree.prefix
    # r-prefix: use default handler
    if x == 'r':
        kernel.iteration_ranges_codegen_header(tree, header_code)
        header_code.splice(code)
        code._lines = header_code._lines
        return

    prefix_block_name = f"{x.upper()}BLOCK"

    # Scalar-odometer axes: tile-1 strided free axes denied a register tensor-dim
    # slot in _apply_linearize. Emit as pure scalar offsets (real_block==1), NOT
    # tiles/aranges, else the register tile inflates to a rank-N shape BiShengIR
    # cannot collapse (~40x slowdown + UB overflow).
    _scalar_odo_names = getattr(tree, '_npu_scalar_odometer', None) or set()

    def _node_block_name(node):
        return prefix_block_name

    # pre_loop_code: loop-invariant defs (real_block_x0, x0_blocks depend only on
    # x0numel and XBLOCK) hoisted before the group loop instead of recomputed.
    if not hasattr(tree, 'pre_loop_code'):
        tree.pre_loop_code = IndentedBuffer()
    pre_loop = tree.pre_loop_code

    # XBLOCK allocation priority by INPUT stride (gated). The default width
    # ``XBLOCK // output_divisor`` carves XBLOCK for a contiguous STORE; for a
    # permute+reduce kernel the input read dominates and its contiguous axis isn't
    # the output's (T5 pos_bias sum: x1/heads has input stride 1 but output divisor
    # 256 -> width 1 -> 1/12 MTE efficiency, ~290us). Ordering the free tile nodes
    # by input stride ASCENDING and giving the most-contiguous axis the full run
    # restores the burst (~40us). Falls back to the divisor formula when the gate
    # is off, <2 free nodes, or strides can't be harvested.
    free_tile_nodes = [n for n in tree.nodes.values()
                       if n.name not in tree.tree_node_mapping]
    # Input-stride priority is an ORDERING signal only: it feeds the greedy
    # prefix-fill allocator (below), which emits axes FLAT (real_block == constexpr
    # tile, no inner loop). The old runtime real_block_* denominator chain
    # (xblock_priority_denom) is kept empty for back-compat with its dead branch.
    xblock_priority_denom = {}
    _priority_instride_order = {}  # node.name -> input-stride rank (most-contig first)
    if (
        ncfg.reduce_xblock_by_input_stride
        and getattr(kernel, "inside_reduction", False)
        and len(free_tile_nodes) >= 2
    ):
        in_strides = _npu_harvest_node_input_strides(kernel, free_tile_nodes)
        if in_strides and all(n.name in in_strides for n in free_tile_nodes):
            # Most-contiguous first; ranks are (degree, const) so symbolic strides
            # order correctly under dynamic shape (1 < ks0 < 16*ks0**2).
            ordered = sorted(free_tile_nodes, key=lambda n: in_strides[n.name])
            _priority_instride_order = {n.name: i for i, n in enumerate(ordered)}

    # Default: insertion order (byte-identical). Priority is applied in the greedy
    # allocator, not to the real_block def nodes.
    if xblock_priority_denom:
        _block_def_nodes = sorted(
            (n for n in tree.nodes.values() if n.name not in tree.tree_node_mapping),
            key=lambda n: list(xblock_priority_denom).index(n.name)
            if n.name in xblock_priority_denom else 1 << 30,
        )
    else:
        _block_def_nodes = [n for n in tree.nodes.values()
                            if n.name not in tree.tree_node_mapping]

    # 32-byte tile alignment (config.tile_align, in elements; default 8 = 32B/fp32).
    # Ascend's vector unit / UB operate on 32B blocks, so an innermost contiguous
    # tile whose width isn't a multiple of A leaves a partial trailing block per row
    # (Wc=111 fp32 = 13.875 blocks; 112 = 14, clean). Round real_block UP for the
    # innermost stride-1 (divisor==1) axis only. real_block is at once the ownership
    # stride, block-count divisor, and arange width, so aligning it keeps all three
    # consistent; padded tail lanes are trimmed by the ``index < numel`` mask.
    # Budget stays safe (real_block = min(numel, XBLOCK), XBLOCK a power of two >=A).
    _tile_align = 0
    _align_node_name = None
    if ncfg.tile_align > 0:
        _tile_align = ncfg.tile_align
    # Skip alignment under a non-OUTER reduction: for INNER/DEFAULT the r-axis is the
    # contiguous burst (R0_BLOCK the innermost slot) and the X-tree axis is strided,
    # so aligning it buys no burst but rounds real_block UP, inflating the UB tile for
    # small-XBLOCK configs (layernorm 262->417us). Only OUTER/OUTER_TINY leave an
    # X-axis as the burst, which still needs align8.
    _skip_align_inner_reduction = False
    if getattr(kernel, "inside_reduction", False):
        _feats = getattr(kernel, "features", None)
        if _feats is not None:
            try:
                from torch._inductor.runtime.hints import ReductionHint
                _skip_align_inner_reduction = (
                    _feats.get_reduction_hint()
                    not in (ReductionHint.OUTER, ReductionHint.OUTER_TINY)
                )
            except Exception:
                _skip_align_inner_reduction = False
    if _tile_align > 1 and not _skip_align_inner_reduction:
        # Innermost contiguous axis = the free node with a static divisor == 1.
        for _n in _block_def_nodes:
            if (
                _n.name not in xblock_priority_denom
                and isinstance(_n.divisor, (int, sympy.Integer))
                and int(_n.divisor) == 1
            ):
                # Skip a SUB-BLOCK axis (static length < align): rounding it up is
                # pure padding, and for a broadcast/expand tail (rope freqs x0
                # numel=2) real_block 2->8 makes the store tile 4x too wide and the
                # padded arange collides with the next axis's stride (7->148us).
                # Only align an axis spanning >= one full block; dynamic lengths
                # keep aligning (can't prove sub-block; dynamic-conv relies on it).
                if isinstance(_n.length, (int, sympy.Integer)) and int(_n.length) < _tile_align:
                    continue
                _align_node_name = _n.name
                break

    def _align_rb(node_name, rhs):
        """Round a real_block RHS up to the next multiple of A (constexpr-preserving;
        innermost contiguous axis only)."""
        if _tile_align <= 1 or node_name != _align_node_name:
            return rhs
        A = _tile_align
        return f"((({rhs}) + {A - 1}) // {A}) * {A}"

    # -----------------------------------------------------------------------
    # UNIFY-BLOCK (config.unify_block, default ON). The old dynamic-shape scheme
    # emitted per axis BOTH a runtime real_block_<x> and a constexpr <x>_blk tile,
    # then walked the gap with an inner loop; the two can disagree (real_block=127
    # vs blk=112) -> a misaligned 2-trip loop and a correctness hazard when they
    # round differently (conv+le+relu hw=256 fail). Unify collapses them into ONE
    # constexpr for non-priority free axes (real_block := aligned tile): no inner
    # loop, runtime block count ceil(numel/real_block), tail trimmed by index<numel.
    # Priority-chained and axis-aware axes keep their original paths.
    _unify_block = ncfg.unify_block

    def _is_unify_candidate(node):
        """A free axis eligible for real_block==tile unification: dynamic-length,
        non-priority, non-axis-aware, and either contiguous (divisor hint 1 with a
        known length hint) or dynamic-stride (divisor hint > 1). Static axes are
        already unified (constexpr real_block, no inner loop)."""
        if not _unify_block:
            return False
        if node.name in xblock_priority_denom:
            return False
        if isinstance(node.length, (int, sympy.Integer)):
            return False  # static length: already unified (constexpr real_block)
        # divisor hint
        dh = None
        if isinstance(node.divisor, (int, sympy.Integer)):
            dh = int(node.divisor)
        else:
            try:
                dh = int(V.graph.sizevars.optimization_hint(node.divisor))
            except Exception:
                dh = None
        if dh == 1:
            # contiguous: unify only when a length hint bounds the tile
            try:
                lh = int(V.graph.sizevars.optimization_hint(node.length))
            except Exception:
                lh = None
            return isinstance(lh, int) and lh > 0
        if isinstance(dh, int) and dh > 1 and not isinstance(node.divisor, (int, sympy.Integer)):
            return True
        return False

    _unify_names = {n.name for n in _block_def_nodes
                    if _is_unify_candidate(n) and n.name not in _scalar_odo_names}

    # -----------------------------------------------------------------------
    # TILE ALLOCATION. Default is GREEDY prefix-fill: order free axes most-
    # contiguous-first, walk a running budget so the innermost axis takes the full
    # contiguous run (up to XBLOCK) and each outer axis eats the remainder
    # (tile_i = min(size_i, rem_i), rem_{i+1} = rem_i // tile_i). This keeps a long
    # HBM burst while flattening the loop nest (an equal-block variant shrank the
    # tail 111->16 and tanked bandwidth). Emitted through the unify path as
    # constexpr tiles (real_block == tile), so the kernel is FLAT (no inner loops).
    #
    # BALANCED mode (config.balanced_target, default OFF) instead spreads the budget
    # for equal block-count per axis (tile_i ∝ size_i); a win only when greedy
    # pinned outer axes to 1, at the cost of a fragmented burst.
    #
    # Every tile is a constexpr EXPRESSION IN XBLOCK, so autotune's XBLOCK sweep
    # rescales it. NOTE: bare min()/max() in @triton.jit resolve to tl.min/tl.max
    # (compile error) -> use ternaries. balanced_target, if set, PINS the budget.
    #
    # Legacy fallback (the T-loop split below) is kept for axis-aware tiling
    # (transpose slot map) and any axis missing a size hint.
    _balanced_tile = {}     # node.name -> constexpr tile expr (str)
    _balanced_shared = []   # ordered [(name, expr)] constexprs to emit first
    _bal_target_fixed = ncfg.balanced_target or None
    _bal_budget = str(_bal_target_fixed) if _bal_target_fixed else prefix_block_name
    # Candidate axes: all free axes except scalar-odometer (no register tile).
    # Input-stride priority reduction axes are included (ordered below).
    _bal_nodes = [
        n for n in _block_def_nodes
        if n.name not in _scalar_odo_names
    ]

    def _size_hint(node):
        if isinstance(node.length, (int, sympy.Integer)):
            return int(node.length)
        try:
            h = int(V.graph.sizevars.optimization_hint(node.length))
            return h if h > 0 else None
        except Exception:
            return None

    def _divisor_hint(node):
        if isinstance(node.divisor, (int, sympy.Integer)):
            return int(node.divisor)
        try:
            return int(V.graph.sizevars.optimization_hint(node.divisor))
        except Exception:
            return None

    # Greedy covers every non-priority free axis. A missing size hint just eats
    # the leftover budget (tile = rem); the running budget keeps prod(tile_i)
    # <= XBLOCK, so the arange product can't exceed Triton's max tensor numel.
    _bal_sizes = {n.name: _size_hint(n) for n in _bal_nodes}
    if _bal_nodes:
        pfx = tree.prefix
        # Most-contiguous first: by input-stride rank when the priority gate fired
        # (reduction, >=2 free axes; T5 position-bias 91x), else by output-divisor
        # hint (pointwise). No-hint axes sort last.
        _ordered = sorted(_bal_nodes, key=(
            lambda n: _priority_instride_order.get(n.name, 1 << 30)
        ) if _priority_instride_order else (
            lambda n: _divisor_hint(n) if _divisor_hint(n) is not None else 1 << 30
        ))
        prev_tile_name = None  # tile constexpr of the previous (inner) axis
        prev_rem_ref = None  # rem constexpr of the previous (inner) axis
        for idx, node in enumerate(_ordered):
            s = _bal_sizes[node.name]  # int or None (unknown)
            if idx == 0:
                rem_ref = _bal_budget
            else:
                # rem_i = rem_{i-1} // tile_{i-1}, floored to >=1.
                rem_name = f"{pfx}_g_rem{idx}"
                rem_raw = f"({prev_rem_ref}) // {prev_tile_name}"
                _balanced_shared.append((rem_name, f"({rem_raw}) if ({rem_raw}) > 1 else 1"))
                rem_ref = rem_name
            tile_name = f"{pfx}_g_tile{idx}"
            if s is None:
                # Unknown size: eat all remaining budget (mask trims the tail).
                tile_expr = f"({rem_ref})"
            else:
                # tile_i = min(size_i, rem_i) via ternary (bare min() is tl.min).
                tile_expr = f"({s}) if ({s}) < ({rem_ref}) else ({rem_ref})"
            _balanced_shared.append((tile_name, tile_expr))
            _balanced_tile[node.name] = tile_name
            prev_tile_name = tile_name
            prev_rem_ref = rem_ref

        # Route greedy axes through the unify path so real_block := tile.
        _unify_names |= set(_balanced_tile)

    # Emit the shared greedy constexprs (rem/tile chain) before any tile uses them.
    for _bs_name, _bs_expr in _balanced_shared:
        pre_loop.writeline(f"{_bs_name} : tl.constexpr = {_bs_expr}")

    for node in _block_def_nodes:
        # Scalar odometer axis: no register tile. real_block==1 so the odometer
        # walks one element/block (block count == numel) and the index is a scalar.
        if node.name in _scalar_odo_names:
            if isinstance(node.length, (int, sympy.Integer)):
                pre_loop.writeline(f"{node.name}numel : tl.constexpr = {int(node.length)}")
            pre_loop.writeline(f"real_block_{node.name} : tl.constexpr = 1")
            continue
        # Unify candidates get real_block emitted after the tile is aligned below.
        if node.name in _unify_names:
            if isinstance(node.length, (int, sympy.Integer)):
                pre_loop.writeline(f"{node.name}numel : tl.constexpr = {int(node.length)}")
            continue
        # Under greedy-via-unify every non-scalar-odo free axis is a unify
        # candidate; the only other path was the legacy real_block=numel//divisor
        # split (now removed). Fail loud rather than skip real_block emission.
        raise RuntimeError(
            f"[triton_experimental] free axis {node.name!r} is neither scalar-odometer nor "
            f"a unify candidate; the legacy non-unify real_block/divisor path was "
            f"removed. kernel={getattr(kernel, 'kernel_name', '?')}"
        )

    node_arange_upper = {}
    node_needs_inner_loop = {}
    node_metadata = {}  # Store metadata for padding optimization

    for node in tree.nodes.values():
        if node.name in tree.tree_node_mapping:
            continue
        # Scalar odometer axis: no arange tile (index is a scalar offset).
        if node.name in _scalar_odo_names:
            node_arange_upper[node.name] = "1"
            node_needs_inner_loop[node.name] = False
            continue
        divisor_is_static = isinstance(node.divisor, (int, sympy.Integer))
        length_is_static = isinstance(node.length, (int, sympy.Integer))
        divisor_hint = None
        try:
            divisor_hint = V.graph.sizevars.optimization_hint(node.divisor)
        except Exception:
            pass
        if divisor_is_static:
            divisor_hint = int(node.divisor)
        block_name = _node_block_name(node)

        # Compile-time length hint for dynamic-length axes: clamp the constexpr
        # arange upper bound to it so the per-axis aranges never multiply past
        # Triton's max tensor numel (1048576) for large XBLOCK.
        length_hint = None
        if not length_is_static:
            try:
                length_hint = int(V.graph.sizevars.optimization_hint(node.length))
            except Exception:
                length_hint = None
            if isinstance(length_hint, int) and length_hint <= 0:
                length_hint = None

        def _clamp_to_length(expr):
            # Clamp a constexpr arange upper-bound expression to the axis length:
            # static length uses the constexpr xNnumel; dynamic length uses the
            # compile-time hint. Both keep the result a pure constexpr.
            if length_is_static:
                return f"({expr}) if ({expr}) <= {node.name}numel else {node.name}numel"
            if length_hint is not None:
                return f"({expr}) if ({expr}) <= {length_hint} else {length_hint}"
            return expr

        # Greedy-via-unify (default for every non-axis-aware free axis, including
        # input-stride priority reduction axes): the constexpr tile is a fixed
        # expression in XBLOCK derived from size hints + the greedy budget (see
        # setup above), ordered most-contiguous-first. Fed straight into the
        # unify hoist as real_block; block count is runtime, tail trimmed by the
        # mask. FLAT -- no inner loop. This subsumes the old priority-chain
        # inner-loop split (real_block runtime + xN_blk constexpr + for xNinner):
        # the greedy order gives the smallest-input-stride axis the full run,
        # which was the sole purpose of that chain.
        if node.name in _balanced_tile:
            arange_upper = str(_balanced_tile[node.name])
            needs_inner_loop = False
        # Defensive catch-all. With greedy-via-unify as the default, every
        # non-priority free axis is already handled by the `_balanced_tile`
        # branch above — so this is unreachable in practice. It survives only as
        # a safe fallback (full-XBLOCK single tile, no inner loop) should some
        # future axis slip past it, keeping codegen total. The legacy
        # divisor-hint / inner-loop pointwise branches were removed here.
        else:
            arange_upper = f"real_block_{node.name}" if (length_is_static and divisor_is_static) else block_name
            needs_inner_loop = False

        node_arange_upper[node.name] = arange_upper
        node_needs_inner_loop[node.name] = needs_inner_loop

        # Store metadata for padding optimization below
        node_metadata[node.name] = {
            'block_name': block_name,
            'divisor_hint': divisor_hint,
            'length_hint': length_hint,
            'length_is_static': length_is_static,
            'divisor_is_static': divisor_is_static,
        }

    # Padding: give the smallest-allocation axis a minimum tile of 8, avoiding
    # scalar fp32 load/store degradation (4xfp32 = 128 bits < 256-bit vector).
    # Gate: pad_min_block_to_8 (default ON).
    if ncfg.pad_min_block_to_8:
        import re
        tile_hints = {}
        for name, expr in node_arange_upper.items():
            # Skip priority-chained nodes: their width was sized to MATCH
            # real_block via the input-stride denominator; rebuilding from the raw
            # output divisor inflates the tile above ownership -> the arange
            # overshoots and outputs are redundantly reduced (16x on T5_large _5).
            if name in xblock_priority_denom:
                continue
            # Skip balanced-tile axes (deliberately sized) and scalar-odometer axes
            # (no register tile / no node_metadata entry).
            if name in _balanced_tile:
                continue
            if name in _scalar_odo_names:
                continue
            literals = re.findall(r'\b(\d+)\b', expr)
            if literals:
                tile_hints[name] = min(int(x) for x in literals)

        if tile_hints:
            min_axis = min(tile_hints, key=lambda n: tile_hints[n])
            min_tile = tile_hints[min_axis]

            if min_tile < 8:
                meta = node_metadata[min_axis]
                block_name = meta['block_name']
                divisor_hint = meta['divisor_hint']
                length_hint = meta['length_hint']
                length_is_static = meta['length_is_static']

                # Rebuild the tile expression with min tile = 8.
                if divisor_hint == 1:
                    # Contiguous: min(XBLOCK, length_hint) -> min(XBLOCK, 8).
                    padded_length = 8
                    if length_is_static:
                        # Static length: clamp to xNnumel
                        new_expr = f"({block_name}) if ({block_name}) <= {padded_length} else {padded_length}"
                        node_name = min_axis
                        new_expr = f"({new_expr}) if ({new_expr}) <= {node_name}numel else {node_name}numel"
                    else:
                        # Dynamic length: clamp to max(8, length_hint).
                        final_clamp = max(8, length_hint) if length_hint is not None else 8
                        new_expr = f"({block_name}) if ({block_name}) <= {final_clamp} else {final_clamp}"
                elif divisor_hint and divisor_hint > 1:
                    # Strided: min(XBLOCK // divisor, length_hint) -> min(..., 8).
                    tile_expr = f"(({block_name} // {divisor_hint}) if {block_name} > {divisor_hint} else 1)"
                    padded_length = 8
                    if length_is_static:
                        node_name = min_axis
                        new_expr = f"({tile_expr}) if ({tile_expr}) <= {padded_length} else {padded_length}"
                        new_expr = f"({new_expr}) if ({new_expr}) <= {node_name}numel else {node_name}numel"
                    else:
                        final_clamp = max(8, length_hint) if length_hint is not None else 8
                        new_expr = f"({tile_expr}) if ({tile_expr}) <= {final_clamp} else {final_clamp}"
                else:
                    new_expr = node_arange_upper[min_axis]

                node_arange_upper[min_axis] = new_expr

    # 32-byte alignment (continued): round the innermost contiguous axis's arange
    # width up to match the aligned real_block above; padded lanes masked by
    # index<numel. Idempotent when numel > XBLOCK (width already XBLOCK-aligned).
    if _align_node_name is not None and _align_node_name in node_arange_upper:
        _a_expr = node_arange_upper[_align_node_name]
        node_arange_upper[_align_node_name] = _align_rb(_align_node_name, _a_expr)

    # Hoist ternary arange upper bounds into constexpr names. For unify candidates
    # the hoisted constexpr IS the per-block ownership span, so name it
    # ``real_block_<x>`` directly and force needs_inner_loop False (the single tile
    # covers one block; runtime block count + index<numel handle the tail).
    for name in node_arange_upper:
        expr = node_arange_upper[name]
        if name in _unify_names:
            # expr is a pure constexpr tile; bind as real_block.
            pre_loop.writeline(f"real_block_{name} : tl.constexpr = {expr}")
            node_arange_upper[name] = f"real_block_{name}"
            node_needs_inner_loop[name] = False
            continue
        if "if " in expr or "//" in expr or "*" in expr:
            blk_name = f"{name}_blk"
            pre_loop.writeline(f"{blk_name} : tl.constexpr = {expr}")
            node_arange_upper[name] = blk_name

    # Odometer: per-axis block count + linear-pid decomposition into per-axis
    # offsets, emitted after real_block. odometer_opt (default ON): B1 hoists the
    # cumulative block product and divides pid once (shorter div chain); B2 drops
    # provably single-block axes (static numel==1 -> offset 0, no cumprod term).
    _odo_opt = ncfg.odometer_opt
    _free_nodes_ordered = [n for n in tree.nodes.values()
                           if n.name not in tree.tree_node_mapping]
    _pid = kernel.iteration_ranges_get_pid(tree)

    def _is_singleton(node):
        return _odo_opt and isinstance(node.length, (int, sympy.Integer)) and int(node.length) == 1

    for node in _free_nodes_ordered:
        divisor_is_static = isinstance(node.divisor, (int, sympy.Integer))
        length_is_static = isinstance(node.length, (int, sympy.Integer))
        blocks_is_constexpr = length_is_static and divisor_is_static
        if blocks_is_constexpr:
            pre_loop.writeline(f"{node.name}_blocks : tl.constexpr = ({node.name}numel + real_block_{node.name} - 1) // real_block_{node.name}")  # noqa: B950
        else:
            pre_loop.writeline(f"{node.name}_blocks = ({node.name}numel + real_block_{node.name} - 1) // real_block_{node.name}")

    if _odo_opt:
        # B1 + B2: cumulative block counts of prior axes (outer_blocks first, then
        # earlier free axes), singleton (numel==1) axes excluded.
        _cum_terms = list(outer_blocks) if outer_blocks else []
        for idx, node in enumerate(_free_nodes_ordered):
            if _is_singleton(node):
                header_code.writeline(f"{node.name}offset = 0")
                continue
            line = f"{node.name}offset = {_pid}"
            if _cum_terms:
                cum_name = f"{tree.prefix}_cumblk_{idx}"
                pre_loop.writeline(f"{cum_name} = " + " * ".join(_cum_terms))
                line += f" // {cum_name}"
            line += f" % {node.name}_blocks * real_block_{node.name}"
            header_code.writeline(line)
            _cum_terms.append(f"{node.name}_blocks")
    else:
        has_offset = list(outer_blocks) if outer_blocks else []
        for node in _free_nodes_ordered:
            line = f"{node.name}offset = {_pid}"
            if len(has_offset) > 0:
                for name in has_offset:
                    line += f" // {name}"
            has_offset.append(f"{node.name}_blocks")
            line += f" % {node.name}_blocks * real_block_{node.name}"
            header_code.writeline(line)

    # Store on tree so _apply_linearize / dense_size_list_npu can use the
    # constexpr-safe block size per node (for tl.full shape generation).
    if not hasattr(tree, 'node_block_constexpr'):
        tree.node_block_constexpr = {}
    tree.node_block_constexpr.update(node_arange_upper)

    for node in tree.nodes.values():
        if node.name in tree.tree_node_mapping:
            continue
        # Scalar odometer axis: the odometer already emitted ``{name}offset``
        # (real_block==1 -> one element per block). Bind the index to that
        # scalar directly -- NO tl.arange, NO tensor-dim broadcast slot (the axis
        # is absent from var_tensor_dims). It broadcasts against the rank-N tiled
        # axes exactly like the validated hand-written scalar-offset form.
        if node.name in _scalar_odo_names:
            header_code.writeline(f"{node.name}index = {node.name}offset")
            header_code.writeline(f"{node.name} = {node.name}index")
            header_code.writeline(f"{node.name}mask = {node.name}offset < {node.name}numel")
            continue
        arange_upper = node_arange_upper[node.name]
        needs_inner_loop = node_needs_inner_loop[node.name]
        if needs_inner_loop:
            continue
        if tree.is_loop:
            header_code.writeline(f"{node.name} = {node.name}offset + {node.name}base")
        elif tree.grid_dim is None:
            header_code.writeline(f"{node.name} = tl.arange(0, {arange_upper})")
            header_code.writeline(f"{node.name}offset = 0")
        else:
            if tree.tensor_dim is not None:
                size = kernel.indexing_size_str(tree.var_tensor_dims[node.name])
                line = f"{node.name}offset + tl.arange(0, {arange_upper}){size}"
            else:
                line = kernel.iteration_ranges_scalar_code(tree, f"{node.name}offset")
            header_code.writeline(f"{node.name}index = {line}")
            header_code.writeline(f"{node.name} = {node.name}index")
        header_code.writeline(f"{node.name}mask = {_mask_cmp_lhs(f'{node.name}index')} < {node.name}numel")

    # Every free axis is now tiled flat (greedy-via-unify / scalar-odometer /
    # static-constexpr): needs_inner_loop is False for all of them, so no axis
    # ever emits a ``for <x>inner in range(...)`` register sub-loop. Assert that
    # invariant rather than carrying the dead inner-loop emitter it once fed.
    if any(
        node_needs_inner_loop.get(node.name, True)
        for node in tree.nodes.values()
        if node.name not in tree.tree_node_mapping
    ):
        raise RuntimeError("[triton_experimental] inner-loop tile split is no longer emitted; a free axis requested one")

    _free_mask_parts = [f'{node.name}mask' for node in tree.nodes.values() if node.name not in tree.tree_node_mapping]
    if _free_mask_parts:
        header_code.writeline(f"{x}mask = {' & '.join(_free_mask_parts)}")

    for node_name, tree_expr in _ordered_mapping_items(tree.tree_node_mapping):
        expr_str, iter_vars = _render_mapping_expr(kernel, tree, tree_expr)
        mask_str = ' & '.join([f'{var}mask' for var in iter_vars])
        header_code.writeline(f"{node_name} = {expr_str}")
        if mask_str:
            header_code.writeline(f"{node_name}mask = {mask_str}")

    header_code.splice(code)
    code._lines = header_code._lines
