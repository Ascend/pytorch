"""as_strided graph rewrite: known layouts become permute/expand before IR build.

Generic Path C AsStrided OPP selects .o but Execute is unstable; known layouts map to
Transpose / Expand which are already green. Layouts this rewrite declines keep the
lowering collector (``as_strided_concretize_arg_hook`` in ``rewrites/arg.py``).

The rewrite self-registers into the framework list at import time.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node

from fxrt.view_offset import (
    ViewOffsetSemantics, explicit_offset, has_zero_storage_origin, view_offset_semantics,
)
from fxrt.gert_backend.rewrites._common import concrete_int, concretize_shape_dim, example_of
from fxrt.gert_backend.rewrites._framework import register_converter_graph_rewrite


def _as_strided_call_targets():
    """Targets that lower to Op.as_strided_view."""
    aten = torch.ops.aten
    return (
        torch.as_strided,
        getattr(aten, "as_strided", None),
        getattr(aten.as_strided, "default", None),
    )


def _fx_node_example(node: Any) -> Any:
    """Best-effort FakeTensor / example from FX node meta."""
    if not isinstance(node, Node):
        return None
    return example_of(node)


def _concrete_view_meta(x_ex: Any) -> Optional[Tuple[List[Any], List[Any]]]:
    """Concretized (shape, stride) lists from a tensor example; None when unavailable.

    None when the example is not a tensor, sits on a nonzero storage offset,
    or any dim fails to concretize.
    """
    if x_ex is None or not isinstance(x_ex, (torch.Tensor, FakeTensor)):
        return None
    try:
        if x_ex.storage_offset() != 0:
            return None
        shape = [concretize_shape_dim(dim) for dim in x_ex.shape]
        stride = [concretize_shape_dim(dim) for dim in x_ex.stride()]
    except (TypeError, RuntimeError):
        return None
    return shape, stride


def _resolve_fx_int_sequence(arg: Any) -> Optional[List[Any]]:
    """Resolve FX size/stride arg to a sequence (may still hold SymInt)."""
    if isinstance(arg, (list, tuple)):
        return list(arg)
    if isinstance(arg, torch.Size):
        return list(arg)
    if isinstance(arg, Node):
        example = example_of(arg)
        if isinstance(example, (list, tuple, torch.Size)):
            return list(example)
    return None


def _all_concrete_nonneg_ints(seq: List[Any]) -> Optional[List[int]]:
    """Return concrete ints when every element concretizes to int >= 0."""
    out: List[int] = []
    for item in seq:
        val = concretize_shape_dim(item)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            return None
        out.append(val)
    return out


def _is_concrete_int_ge(val: Any, lo: int) -> bool:
    """True when val is a real (non-bool) int >= lo."""
    return isinstance(val, int) and not isinstance(val, bool) and val >= lo


def _squeeze_to_matrix_meta(x_shape: List[Any], x_stride: List[Any]):
    """Return (shape2d, stride2d, squeeze_dim) when a size-1 dim hides a matrix.

    ``(150, 1, 32)`` with strides ``(32, 32, 1)`` is a ``(150, 32)`` contiguous
    matrix with a unit dim wedged in the middle; squeezing that dim yields the
    matrix the transpose patterns expect. None when no single unit dim can be
    removed to leave a rank-2 view.
    """
    if len(x_shape) != len(x_stride) or len(x_shape) < 3:
        return None
    unit_dims = [i for i, dim in enumerate(x_shape) if dim == 1]
    if len(unit_dims) != 1:
        return None
    drop = unit_dims[0]
    shape2d = [dim for i, dim in enumerate(x_shape) if i != drop]
    stride2d = [dim for i, dim in enumerate(x_stride) if i != drop]
    if len(shape2d) != 2:
        return None
    return shape2d, stride2d, drop


def _is_transpose_expand_as_strided(size_seq: List[Any], stride_seq: List[Any], x) -> bool:
    """True for size (B,H,W) strides (0,1,H) == contiguous (W,H).T.expand(B,...).

    Batch dim may remain SymInt; H/W and strides must be concrete. The rewrite
    emits ``x.permute(1,0).expand(B,H,W)``, which needs the base to be a
    rank-2 ``(W,H)`` contiguous matrix. A rank-3 base is accepted only when a
    single unit dim hides that matrix (squeezed by the builder), otherwise
    ``permute([1,0])`` would raise "permute a tensor of rank 3".
    """
    if len(size_seq) != 3 or len(stride_seq) != 3:
        return False
    height = concretize_shape_dim(size_seq[1])
    width = concretize_shape_dim(size_seq[2])
    if not _is_concrete_int_ge(height, 1) or not _is_concrete_int_ge(width, 1):
        return False
    stride = _all_concrete_nonneg_ints(stride_seq)
    if stride != [0, 1, height]:
        return False
    meta = _concrete_view_meta(_fx_node_example(x))
    if meta is None:
        return False
    x_shape, x_stride = meta
    if x_shape == [width, height] and x_stride == [height, 1]:
        return True
    squeezed = _squeeze_to_matrix_meta(x_shape, x_stride)
    if squeezed is None:
        return False
    shape2d, stride2d, _ = squeezed
    return shape2d == [width, height] and stride2d == [height, 1]


def _is_matrix_transpose_as_strided(size_seq: List[Any], stride_seq: List[Any], x) -> bool:
    """True for size (R,C) strides (1,R) == contiguous (C,R).T."""
    if len(size_seq) != 2 or len(stride_seq) != 2:
        return False
    size = _all_concrete_nonneg_ints(size_seq)
    stride = _all_concrete_nonneg_ints(stride_seq)
    if size is None or stride is None:
        return False
    rows, cols = size
    meta = _concrete_view_meta(_fx_node_example(x))
    if meta is None:
        return False
    source_shape = _all_concrete_nonneg_ints(meta[0])
    source_stride = _all_concrete_nonneg_ints(meta[1])
    return (stride == [1, rows] and rows >= 1
            and source_shape == [cols, rows] and source_stride == [rows, 1])


def _is_leading_broadcast_as_strided(size_seq: List[Any], stride_seq: List[Any], x) -> bool:
    """True for a contiguous matrix broadcast through a leading zero-stride dim.

    A layout such as ``size=(B,H,W), stride=(0,W,1)`` is exactly
    ``x.unsqueeze(0).expand(B,H,W)`` when ``x`` is a contiguous ``(H,W)``
    matrix.  Keeping this as an explicit expand avoids the unstable generic
    AsStrided Path C implementation while preserving the alias/broadcast
    semantics.
    """
    if len(size_seq) != 3 or len(stride_seq) != 3:
        return False
    batch = concretize_shape_dim(size_seq[0])
    height = concretize_shape_dim(size_seq[1])
    width = concretize_shape_dim(size_seq[2])
    stride = _all_concrete_nonneg_ints(stride_seq)
    if not _is_concrete_int_ge(batch, -1):
        return False
    if not _is_concrete_int_ge(height, 1) or not _is_concrete_int_ge(width, 1):
        return False
    if stride != [0, width, 1]:
        return False
    meta = _concrete_view_meta(_fx_node_example(x))
    if meta is None:
        return False
    x_shape, x_stride = meta
    return x_shape == [height, width] and x_stride == [width, 1]


def _is_identity_as_strided(size_seq: List[Any], stride_seq: List[Any], x) -> bool:
    """True when as_strided repeats the input tensor's existing view metadata."""
    x_ex = _fx_node_example(x)
    if x_ex is None or not isinstance(x_ex, (torch.Tensor, FakeTensor)):
        return False
    if len(size_seq) != x_ex.ndim or len(stride_seq) != x_ex.ndim:
        return False
    try:
        if x_ex.storage_offset() != 0:
            return False
        target_size = [concretize_shape_dim(dim) for dim in size_seq]
        target_stride = [concretize_shape_dim(dim) for dim in stride_seq]
        x_size = [concretize_shape_dim(dim) for dim in x_ex.shape]
        x_stride = [concretize_shape_dim(dim) for dim in x_ex.stride()]
    except (TypeError, RuntimeError):
        return False
    if target_size != x_size:
        return False
    # Strides on singleton dimensions do not affect addressing and may differ
    # after Inductor has reconstructed an otherwise identical view.
    return all(size == 1 or target == current
               for size, target, current in zip(target_size, target_stride, x_stride))


def _as_strided_offset_or_none(node: Node) -> Optional[int]:
    """Return concrete storage_offset, or None if missing/invalid."""
    offset = explicit_offset(node)
    if offset is None:
        return 0
    off = concretize_shape_dim(offset)
    if not isinstance(off, int) or isinstance(off, bool):
        return None
    return off


def _concrete_expand_size(size_seq: List[Any]) -> Optional[List[int]]:
    """Concretize expand dims to an ABI0 int_list; None if any dim stays symbolic.

    Dynamic batch keeps dim -1; never int(SymInt).
    """
    expand_size: List[int] = []
    for dim in size_seq:
        val = concretize_shape_dim(dim)
        if not _is_concrete_int_ge(val, -1):
            return None
        expand_size.append(val)
    return expand_size


def _set_view_meta(fx_node: Node, example: Any) -> None:
    """Stamp example_value/val meta on a freshly built view node."""
    fx_node.meta["example_value"] = example
    fx_node.meta["val"] = example


def _copy_view_meta(repl: Node, node: Node) -> None:
    """Propagate example_value/val meta from the rewritten node when present."""
    if "example_value" in node.meta:
        repl.meta["example_value"] = node.meta["example_value"]
    if "val" in node.meta:
        repl.meta["val"] = node.meta["val"]


def _example_via(x, method: str, *args: Any) -> Any:
    """Run the same view op on the input's fake example; None if unavailable."""
    x_ex = _fx_node_example(x)
    if x_ex is None or not hasattr(x_ex, method):
        return None
    return getattr(x_ex, method)(*args)


def _backfill_expand_meta(repl: Node, node: Node, perm: Node, expand_size: List[int]) -> None:
    """Derive expand example meta from perm when the rewritten node had no val meta."""
    if "val" in node.meta or perm.meta.get("example_value") is None:
        return
    try:
        _set_view_meta(repl, perm.meta["example_value"].expand(expand_size))
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _build_leading_broadcast_rewrite(graph, node: Node, x, size_seq: List[Any]):
    """Build the x.unsqueeze(0).expand(B,H,W) replacement; None if dims stay symbolic."""
    expand_size = _concrete_expand_size(size_seq)
    if expand_size is None:
        return None
    unsqueezed = graph.call_function(torch.ops.aten.unsqueeze.default, args=(x, 0))
    unsq_ex = _example_via(x, "unsqueeze", 0)
    if unsq_ex is not None:
        _set_view_meta(unsqueezed, unsq_ex)
    repl = graph.call_function(torch.ops.aten.expand.default, args=(unsqueezed, expand_size))
    _copy_view_meta(repl, node)
    return repl


def _build_transpose_expand_rewrite(graph, node: Node, x, size_seq: List[Any]):
    """Build the x.permute(1,0).expand(...) replacement; None if dims stay symbolic.

    When the base carries a single unit dim around the matrix, squeeze it first so
    the trailing permute(1,0) sees the rank-2 matrix.
    """
    expand_size = _concrete_expand_size(size_seq)
    if expand_size is None:
        return None
    base = x
    meta = _concrete_view_meta(_fx_node_example(x))
    if meta is not None:
        squeezed = _squeeze_to_matrix_meta(meta[0], meta[1])
        if squeezed is not None:
            _, _, drop = squeezed
            base = graph.call_function(torch.ops.aten.squeeze.dim, args=(x, drop))
            squeeze_ex = _example_via(x, "squeeze", drop)
            if squeeze_ex is not None:
                _set_view_meta(base, squeeze_ex)
    perm = graph.call_function(torch.ops.aten.permute.default, args=(base, [1, 0]))
    perm_ex = _example_via(base, "permute", 1, 0)
    if perm_ex is not None:
        _set_view_meta(perm, perm_ex)
    repl = graph.call_function(torch.ops.aten.expand.default, args=(perm, expand_size))
    _copy_view_meta(repl, node)
    _backfill_expand_meta(repl, node, perm, expand_size)
    return repl


def _build_matrix_transpose_rewrite(graph, x):
    """Build the x.permute(1,0) replacement."""
    repl = graph.call_function(torch.ops.aten.permute.default, args=(x, [1, 0]))
    perm_ex = _example_via(x, "permute", 1, 0)
    if perm_ex is not None:
        _set_view_meta(repl, perm_ex)
    return repl


def _try_build_as_strided_rewrite(graph, node: Node, x, size_seq: List[Any],
                                  stride_seq: List[Any]):
    """Insert permute/expand rewrite before node; return replacement Node or None."""
    if _is_identity_as_strided(size_seq, stride_seq, x):
        return x
    if _is_leading_broadcast_as_strided(size_seq, stride_seq, x):
        return _build_leading_broadcast_rewrite(graph, node, x, size_seq)
    if _is_transpose_expand_as_strided(size_seq, stride_seq, x):
        return _build_transpose_expand_rewrite(graph, node, x, size_seq)
    if _is_matrix_transpose_as_strided(size_seq, stride_seq, x):
        return _build_matrix_transpose_rewrite(graph, x)
    return None


def _rewrite_as_strided_known_patterns_(gm: GraphModule) -> None:
    """Rewrite selective as_strided patterns to permute/expand (Path C stable).

    Generic Path C AsStrided OPP selects .o but Execute is unstable; known layouts
    map to Transpose / Expand which are already green.
    """
    targets = {t for t in _as_strided_call_targets() if t is not None}
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        # The shared layout prelude deliberately emits metadata views. Rewriting
        # them to device Transpose/Expand here would bypass the C++ edge planner.
        if node.meta.get("_fxrt_layout_kind") is not None:
            continue
        if node.op != "call_function" or node.target not in targets or len(node.args) < 3:
            continue
        size_seq = _resolve_fx_int_sequence(node.args[1])
        stride_seq = _resolve_fx_int_sequence(node.args[2])
        if size_seq is None or stride_seq is None:
            continue
        offset = _as_strided_offset_or_none(node)
        if offset is None or offset != 0:
            continue
        semantics = view_offset_semantics(node)
        if semantics == ViewOffsetSemantics.FEED_RELATIVE:
            # This delta names the feed origin, which need not be the current
            # input view's origin. Retain its explicit geometry for lowering.
            continue
        if (semantics == ViewOffsetSemantics.ABSOLUTE
                and not has_zero_storage_origin(node.args[0])):
            # A sampled feed offset of zero does not prove a cached graph's
            # live origin. Permute/expand inherit it; absolute as_strided does not.
            continue
        source_meta = _fx_node_example(node.args[0])
        # These pattern replacements assume that offset 0 is the source's
        # actual storage origin. A cropped feed uses a different data_ptr.
        if source_meta is None or concrete_int(source_meta.storage_offset()) != 0:
            continue
        with graph.inserting_before(node):
            repl = _try_build_as_strided_rewrite(graph, node, node.args[0], size_seq, stride_seq)
        if repl is None:
            continue
        node.replace_all_uses_with(repl)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()


# Known as_strided layouts become permute/expand before IR build. Layouts this
# rewrite declines keep the lowering collector.
register_converter_graph_rewrite(_rewrite_as_strided_known_patterns_)
