# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""NPU-specific FX graph passes for the triton_experimental backend.

Two self-contained passes, each gated behind a ``config`` flag and installed by
``overrides.apply_npu_overrides`` at backend activation:

* ``_elide_int_float_int_roundtrip_pass`` -- a post_grad custom pass that deletes
  ``int -> float -> ... -> int`` round-trip casts (installed via
  ``_install_elide_int_float_int_pass``).
* ``_npu_fold_max1`` / ``_install_fold_max1_in_loop_merge`` -- a
  ``SizeVarAllocator._simplify_loops_impl`` monkeypatch that folds ``Max(1, e)``
  strides so contiguous conv-output axes merge.
"""
import logging

import sympy
import torch

from . import config as ncfg

log = logging.getLogger("torch._inductor")


def _elide_int_float_int_roundtrip_pass(graph):
    """Rewrite ``int_x -> float -> (views/scatter) -> int`` back to a pure int view
    of the original int source, deleting the round-trip cast.

    Pattern (DIFM sparse-column prologue): the model stuffs integer feature-ids into
    a float X_mat via slice_scatter then pulls them back with .long(), so
    ``X[:, k:k+1].to(i64)`` provably equals ``arg1_1[:, k:k+1]``. Inductor's
    pointless_convert can't fold it (needs an all-float chain with adjacent converts;
    here the base is int and a slice_scatter/slice sits between the casts).

    We trace each ``convert_element_type -> <int>`` node backwards through value-
    preserving ops, carrying a per-dim read window (slice shifts it, select
    re-inserts a pinned axis, squeeze/unsqueeze/view remap size-1 axes 1:1 else bail,
    copy/clone pass through, slice_scatter maps onto source or base). If the window
    lands on an int source of the same dtype having crossed >=1 int->float cast, the
    convert is replaced by slice(s) + reshape of that int source. For DIFM this
    deletes all 26 column-extract kernels.

    CORRECTNESS: i64->f32->i64 is bit-exact only for |value| < 2**24 (f32 mantissa);
    embedding indices always are, but it's not universally value-preserving, so it's
    gated behind elide_int_float_int (default ON). The source
    int dtype must equal the target int dtype exactly (no width change).
    """
    if not ncfg.elide_int_float_int:
        return

    aten = torch.ops.aten
    prims = torch.ops.prims

    _INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64,
                   torch.uint8, torch.bool)
    _FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    def _val(node):
        return node.meta.get("val", None) if hasattr(node, "meta") else None

    def _dtype(node):
        v = _val(node)
        return getattr(v, "dtype", None) if v is not None else None

    def _size_on(node, dim):
        v = _val(node)
        if v is None or not hasattr(v, "shape"):
            return None
        try:
            return int(v.shape[dim])
        except Exception:
            return None

    def _norm(idx, size):
        """Normalize a (possibly negative / INT64_MAX) slice bound to [0, size]."""
        if idx is None:
            return 0
        idx = int(idx)
        if idx < 0:
            idx += size
        if idx > size:
            idx = size
        if idx < 0:
            idx = 0
        return idx

    def _nonone(shape):
        """Indices of the non-size-1 axes of ``shape`` (row-major order)."""
        return [i for i, s in enumerate(shape) if s != 1]

    def _map_rank_adjust(in_shape, out_shape, windows):
        """Re-express ``windows`` (keyed by OUT-frame dim) into the IN frame for
        a pure rank-adjust op (view/reshape/squeeze/unsqueeze) that only
        inserts or removes size-1 axes.  The non-1 axes must line up 1:1 in
        order (same count, same sizes); otherwise the op merges/splits real
        axes and we cannot model it as a window -> return None.

        Windows landing on a size-1 OUT axis are dropped (a size-1 window is
        always the full extent, so it constrains nothing)."""
        in_nz = _nonone(in_shape)
        out_nz = _nonone(out_shape)
        if len(in_nz) != len(out_nz):
            return None
        for a, b in zip(in_nz, out_nz):
            if in_shape[a] != out_shape[b]:
                return None
        out2in = {ob: ia for ia, ob in zip(in_nz, out_nz)}
        new = {}
        for od, win in windows.items():
            if od not in out2in:
                # window on a size-1 OUT axis -> full extent, nothing to carry.
                continue
            new[out2in[od]] = win
        return new

    def _trace(node, windows, saw_float):
        """Walk value provenance backward.  ``windows`` maps a dim index (in
        ``node``'s current coordinate frame) to a ``(lo, hi)`` half-open window;
        dims absent from the dict span their full extent.  Return
        ``(int_source_node, windows_in_source_frame)`` when ``node`` provably
        comes from an int source of the target dtype after >=1 int->float cast;
        else None.  ``_trace._target`` is the outer convert's dtype."""
        seen = 0
        while node is not None and isinstance(node, torch.fx.Node):
            seen += 1
            if seen > 64:
                return None  # runaway guard
            dt = _dtype(node)
            # Accept: an integer source of the exact target dtype, reached after
            # we have crossed an int->float cast (so this really was a roundtrip).
            if saw_float and dt == _trace._target and dt in _INT_DTYPES:
                return (node, windows)

            if node.op != "call_function":
                return None
            tgt = node.target

            if tgt == prims.convert_element_type.default:
                src = node.args[0]
                out_dt = node.args[1] if len(node.args) > 1 else None
                src_dt = _dtype(src)
                # Only int->float is safe to walk past (the forward half of the
                # round-trip, value-preserving for |v|<2**24). float->int truncates,
                # float->float narrows, int->int changes width -> bail on all.
                if out_dt in _FLOAT_DTYPES and src_dt in _INT_DTYPES:
                    saw_float = True
                    node = src
                    continue
                return None

            if tgt == aten._to_copy.default:
                # Functionalized graphs express the int->float cast as
                # _to_copy(dtype=...). int->float sets the flag; same/no dtype is a
                # passthrough; anything lossy bails like the convert case.
                src = node.args[0]
                cp_dt = node.kwargs.get("dtype", None)
                if cp_dt is None and len(node.args) > 1:
                    cp_dt = node.args[1]
                src_dt = _dtype(src)
                if cp_dt is None or cp_dt == src_dt:
                    node = src
                    continue
                if cp_dt in _FLOAT_DTYPES and src_dt in _INT_DTYPES:
                    saw_float = True
                    node = src
                    continue
                return None

            if tgt == aten.copy.default:
                # copy(dst, src) returns src's values reshaped to dst's shape;
                # shapes match in our pattern, so follow the src operand.
                node = node.args[1]
                continue

            if tgt == aten.clone.default:
                node = node.args[0]
                continue

            if tgt == aten.slice.Tensor:
                s_in = node.args[0]
                s_dim = int(node.args[1]) if len(node.args) > 1 else 0
                sz_in = _size_on(s_in, s_dim)
                if sz_in is None:
                    return None
                s_start = _norm(node.args[2] if len(node.args) > 2 else 0, sz_in)
                s_end = _norm(node.args[3] if len(node.args) > 3 else sz_in, sz_in)
                s_step = int(node.args[4]) if len(node.args) > 4 else 1
                if s_step != 1:
                    return None
                lo, hi = windows.get(s_dim, (0, _size_on(node, s_dim)))
                if lo is None or hi is None:
                    return None
                # shift the window back into the input's coordinate frame.
                lo = s_start + lo
                hi = s_start + hi
                if hi > s_end:
                    return None
                windows = dict(windows)
                windows[s_dim] = (lo, hi)
                node = s_in
                continue

            if tgt == aten.select.int:
                # select(dim, index) drops ``dim``; the OUT frame has one fewer
                # axis. Re-insert it: shift every OUT dim >= sel_dim up by one,
                # then pin sel_dim to the single row [index:index+1].
                s_in = node.args[0]
                sel_dim = int(node.args[1]) if len(node.args) > 1 else 0
                sz_in = _size_on(s_in, sel_dim)
                if sz_in is None:
                    return None
                idx = _norm(int(node.args[2]) if len(node.args) > 2 else 0, sz_in)
                shifted = {}
                for od, win in windows.items():
                    shifted[od + 1 if od >= sel_dim else od] = win
                shifted[sel_dim] = (idx, idx + 1)
                windows = shifted
                node = s_in
                continue

            if tgt == aten.slice_scatter.default:
                base, src = node.args[0], node.args[1]
                sc_dim = int(node.args[2]) if len(node.args) > 2 else 0
                sz = _size_on(node, sc_dim)
                if sz is None:
                    return None
                sc_start = _norm(node.args[3] if len(node.args) > 3 else 0, sz)
                sc_end = _norm(node.args[4] if len(node.args) > 4 else sz, sz)
                sc_step = int(node.args[5]) if len(node.args) > 5 else 1
                if sc_step != 1:
                    return None
                lo, hi = windows.get(sc_dim, (0, sz))
                if lo is None or hi is None:
                    return None
                if lo >= sc_start and hi <= sc_end:
                    # fully inside the scattered region -> src, reindexed.
                    windows = dict(windows)
                    windows[sc_dim] = (lo - sc_start, hi - sc_start)
                    node = src
                    continue
                if hi <= sc_start or lo >= sc_end:
                    # fully outside -> base, coordinates unchanged.
                    node = base
                    continue
                return None  # straddles the boundary: bail

            if tgt in (aten.view.default, aten.reshape.default,
                       aten.squeeze.default, aten.squeeze.dim,
                       aten.unsqueeze.default):
                # Pure rank-adjust: safe only when it just inserts/removes
                # size-1 axes (non-1 axes line up 1:1). Anything that merges or
                # splits real axes returns None from _map_rank_adjust and bails.
                s_in = node.args[0]
                iv = _val(s_in)
                ov = _val(node)
                if iv is None or ov is None or not hasattr(iv, "shape"):
                    return None
                mapped = _map_rank_adjust(tuple(iv.shape), tuple(ov.shape), windows)
                if mapped is None:
                    return None
                windows = mapped
                node = s_in
                continue

            return None
        return None

    rewritten = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != prims.convert_element_type.default:
            continue
        out_dt = node.args[1] if len(node.args) > 1 else None
        if out_dt not in _INT_DTYPES:
            continue
        src = node.args[0]
        if not isinstance(src, torch.fx.Node):
            continue
        v = _val(node)
        if v is None or not hasattr(v, "shape") or v.ndim == 0:
            continue

        # Trace value provenance with an empty window set (full extent on every
        # dim of the convert's own output); the tracer fills in windows as it
        # crosses slices/selects/scatters.
        _trace._target = out_dt
        res = _trace(src, {}, saw_float=False)
        if res is None:
            continue
        int_src, windows = res
        iv = _val(int_src)
        if iv is None or not hasattr(iv, "shape"):
            continue
        src_rank = iv.ndim

        # Build the replacement: slice the int source on every constrained dim,
        # then reshape to the convert's exact output shape. Element order is
        # preserved through every op we walked, so the reshape is a pure view of
        # the selected elements. Skip windows that already span the full extent.
        made = []
        try:
            with graph.inserting_before(node):
                cur = int_src
                for sd in range(src_rank):
                    win = windows.get(sd)
                    if win is None:
                        continue
                    lo, hi = win
                    full = _size_on(int_src, sd)
                    if lo == 0 and (full is None or hi == full):
                        continue
                    nxt = graph.call_function(
                        aten.slice.Tensor, (cur, sd, lo, hi, 1)
                    )
                    made.append(nxt)
                    cur = nxt

                out_shape = tuple(int(s) for s in v.shape)
                cur_v = _val(cur)
                need_reshape = (
                    cur_v is None
                    or tuple(int(s) for s in cur_v.shape) != out_shape
                )
                if need_reshape:
                    nxt = graph.call_function(
                        aten.reshape.default, (cur, list(out_shape))
                    )
                    made.append(nxt)
                    cur = nxt
                repl = cur

            # Propagate FakeTensor meta along the newly built chain so lowering
            # sees the correct i64 shapes.
            if made:
                fm = getattr(iv, "fake_mode", None)
                if fm is None:
                    raise RuntimeError("no fake_mode on int source")
                with fm:
                    prev_v = iv
                    for m in made:
                        if m.target == aten.slice.Tensor:
                            _, sd, lo, hi, st = m.args
                            prev_v = torch.ops.aten.slice.Tensor(prev_v, sd, lo, hi, st)
                        else:  # reshape
                            prev_v = torch.ops.aten.reshape.default(prev_v, m.args[1])
                        m.meta["val"] = prev_v

            # sanity: replacement dtype/shape must match the node we replace.
            rv = _val(repl)
            nv = _val(node)
            if rv is not None and nv is not None:
                if rv.dtype != nv.dtype or tuple(rv.shape) != tuple(nv.shape):
                    log.debug("[NPU] int-float-int elide skip (meta mismatch): %s %s/%s vs %s/%s",
                              node.name, tuple(nv.shape), nv.dtype, tuple(rv.shape), rv.dtype)
                    for m in reversed(made):
                        graph.erase_node(m)
                    continue
        except Exception as e:
            log.debug("[NPU] int-float-int elide meta propagation failed: %r", e)  # noqa: G200
            for m in reversed(made):
                try:
                    graph.erase_node(m)
                except Exception:
                    pass
            continue

        node.replace_all_uses_with(repl)
        graph.erase_node(node)
        rewritten += 1
        log.debug("[NPU] elided int->float->int roundtrip: %s -> %s (src %s, windows %s)",
                  node.name, repl.name, int_src.name, windows)

    if rewritten:
        graph.eliminate_dead_code()
        graph.lint()
        log.debug("[NPU] int-float-int roundtrip elide: rewrote %s site(s)", rewritten)


def _install_elide_int_float_int_pass():
    """Register _elide_int_float_int_roundtrip_pass as an inductor post_grad
    custom pass, composing with any pass already set."""
    if not ncfg.elide_int_float_int:
        return
    from torch._inductor import config as inductor_config

    prev = inductor_config.post_grad_custom_post_pass

    def _composed(graph):
        if prev is not None:
            prev(graph)
        _elide_int_float_int_roundtrip_pass(graph)

    inductor_config.post_grad_custom_post_pass = _composed
    log.debug("[NPU] installed int->float->int roundtrip elide post_grad pass")


def _npu_fold_max1(expr):
    """Structurally rewrite ``Max(1, e) -> e`` throughout a sympy expression.

    torch's Max is NOT a subclass of sympy.Max -- match it by class name.
    make_contiguous_strides_for wraps stride atoms as Max(1, dim); a bare symbol
    auto-folds but Max(1, FloorDiv(s0-3,2)+1) (conv output H/W) has unknown sign
    and stays wrapped, blocking contiguous-axis merging. Exact for any non-empty
    tensor (every dim >= 1); used only in the merge DECISION, never in returned
    index formulas.
    """
    if not isinstance(expr, sympy.Basic):
        return expr

    def _is_max1(node):
        return (
            type(node).__name__ == "Max"
            and getattr(node, "is_Function", False)
            and len(node.args) == 2
            and any(a == 1 for a in node.args)
        )

    def _drop_one(node):
        rest = [a for a in node.args if a != 1]
        return rest[0] if rest else node

    try:
        return expr.replace(_is_max1, _drop_one)
    except Exception:
        return expr


def _install_fold_max1_in_loop_merge():
    """Make contiguous conv-output axes (H/W with ``Max(1, FloorDiv)`` strides)
    merge in ``SizeVarAllocator._simplify_loops_impl``.

    can_merge_dims tests ``Wc == Max(1, Wc)`` for a contiguous conv output, which
    sympy can't prove, so no spatial axes merge and the kernel degrades to a 4-axis
    strided-burst loop. Folding Max(1,e)->e in COPIES of strides/sizes/formulas
    used only by the merge test lets these axes collapse to one stride-1 sweep;
    the returned sizes/reindex/prune use the ORIGINAL unfolded sizes, so nothing
    folded escapes into emitted indexing (safe for reduction bodies).
    """
    if not ncfg.fold_max1_in_loop_merge:
        return

    from torch._inductor.sizevars import SizeVarAllocator
    from torch._inductor.utils import sympy_subs as _sympy_subs
    try:
        from torch._inductor.sizevars import sympy_index_symbol as _sym
    except Exception:
        from torch._inductor.utils import sympy_index_symbol as _sym

    if getattr(SizeVarAllocator, "_npu_fold_max1_installed", False):
        return

    def _patched_simplify_loops_impl(self, index_vars, sizes, index_formulas):
        sizes = [self.simplify(s) for s in sizes]
        strides = []
        for formula in index_formulas:
            if isinstance(formula, sympy.Expr):
                strides.append(self.stride_vars(formula, index_vars))
            else:
                strides.append([0] * len(index_vars))
        if len(sizes) != len(strides[0]):
            raise RuntimeError(f"[triton_experimental] sizes/strides length mismatch: {len(sizes)} vs {len(strides[0])}")

        # Unit dimensions carry no information; collapse them to a None marker
        # so they are dropped from the returned shape but still tracked here.
        sizes = [None if s == 1 else s for s in sizes]

        # Folded shadow copies used ONLY by the merge test below. The real
        # `sizes` (with Max(1,·)) drive reindex/prune and the return value.
        fsizes = [None if s is None else _npu_fold_max1(s) for s in sizes]
        fstrides = [[_npu_fold_max1(st) for st in row] for row in strides]
        fformulas = [_npu_fold_max1(f) for f in index_formulas]

        def can_merge_dims(a, b):
            # Dims a and b fold together only when EVERY formula agrees: the
            # inner stride must line up with the outer stride*size, and the
            # linearized substitution must match the split one.
            for k in range(len(strides)):
                aligned = self.simplify(fstrides[k][a] * fsizes[a]) == self.simplify(
                    fstrides[k][b]
                )
                if not aligned:
                    return False
                va = index_vars[a]
                vb = index_vars[b]
                m1 = _sym("_merge_tester1")
                m2 = _sym("_merge_tester2")
                linear = _sympy_subs(fformulas[k], {va: m1 * fsizes[a], vb: m2})
                split = _sympy_subs(fformulas[k], {va: 0, vb: (m1 + m2)})
                if self.simplify(linear) != self.simplify(split):
                    return False
            return True

        ndim = len(sizes)
        merged = True
        while merged:
            merged = False
            for a in range(ndim - 1, -1, -1):
                for b in range(ndim - 1, -1, -1):
                    if a == b or sizes[a] is None or sizes[b] is None:
                        continue
                    if can_merge_dims(a, b):
                        merged = True
                        sizes[a] = sizes[a] * sizes[b]
                        sizes[b] = None
                        # keep folded shadow in step so later tests stay consistent
                        fsizes[a] = _npu_fold_max1(sizes[a])
                        fsizes[b] = None

        _reindex_sentinel = object()

        def reindex(index):
            stream = iter(index)
            rebuilt = []
            for size in sizes:
                if size is None:
                    rebuilt.append(sympy.S.Zero)
                else:
                    rebuilt.append(next(stream))
            if next(stream, _reindex_sentinel) is not _reindex_sentinel:
                raise RuntimeError("[triton_experimental] reindex did not consume all index entries")
            return rebuilt

        def prune(index):
            if len(index) != len(sizes):
                raise RuntimeError(f"[triton_experimental] prune index length mismatch: {len(index)} vs {len(sizes)}")
            return [entry for entry, size in zip(index, sizes) if size is not None]

        return [s for s in sizes if s is not None], reindex, prune

    SizeVarAllocator._simplify_loops_impl = _patched_simplify_loops_impl
    SizeVarAllocator._npu_fold_max1_installed = True
    log.debug("[NPU] installed Max(1,·) loop-merge fold (comparison-only)")


def _disable_addmm_fusion_pass():
    # Disable ONLY the post_grad add+mm -> addmm fusion. should_prefer_unfused_addmm
    # returns False for non-GPU devices (is_gpu excludes npu), so on NPU the fusion
    # fires unconditionally. extra_check is captured by value at import time, so
    # patching the module name is useless; instead neuter the extra_check of the
    # already-registered fusion entries (handler named `addmm`) -- pattern_matcher
    # gates on is_match and extra_check, so a False check skips just this fusion.
    if not ncfg.disable_addmm_fusion:
        return
    from torch._inductor.fx_passes import post_grad  # ensure patterns are registered

    patched = 0
    for entries in post_grad.pass_patterns[2].patterns.values():
        for entry in entries:
            handler = getattr(entry, "handler", None)
            if getattr(handler, "__name__", None) == "addmm":
                entry.extra_check = lambda match: False
                patched += 1
    log.debug("[NPU] disabled add+mm->addmm fusion: neutered %s pattern entries", patched)
