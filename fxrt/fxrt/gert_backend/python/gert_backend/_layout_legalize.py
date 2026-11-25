"""Torch-aligned layout contract for the converter FX prelude.

Used only by _fx_to_c_graph; see that module's docstring for why both live here
rather than in torch_npu.

View producers (aten.expand / permute / transpose / t) become as_strided.
Consumers that require a dense StorageShape get a materializing op on that
edge only (expand or permute-from-base). compiled-kernel args keep the view.
Last-two permute into mm/bmm/matmul/addmm stays a view for DualStage Path B.
Non-zero-offset and non-expand as_strided layouts stay on their explicit
materialization path instead of being guessed to be Expand.
Broadcast-weight bmm is rewritten to mm so [B,K,N] Expand is not inserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node

aten = torch.ops.aten

_EXPAND_TARGETS = {
    aten.expand.default,
    aten.expand,
    torch.Tensor.expand,
}

_PERMUTE_FUNC_TARGETS = {
    aten.permute.default,
    aten.permute,
    aten.transpose.int,
    aten.t.default,
    aten.movedim.int,
    aten.movedim.intlist,
    torch.permute,
    torch.transpose,
    torch.t,
    torch.movedim,
}

_PERMUTE_METHOD_NAMES = frozenset({"permute", "transpose", "t", "movedim"})

_AS_STRIDED_FUNC_TARGETS = {
    aten.as_strided.default,
    aten.as_strided,
    torch.as_strided,
}

_BMM_FUNC_TARGETS = {
    aten.bmm.default,
    aten.bmm,
    torch.bmm,
}

_MM_FAMILY_FUNC_TARGETS = {
    aten.mm.default,
    aten.mm,
    torch.mm,
    aten.bmm.default,
    aten.bmm,
    torch.bmm,
    aten.matmul.default,
    aten.matmul,
    torch.matmul,
    aten.addmm.default,
    aten.addmm,
    torch.addmm,
}

_MM_FAMILY_METHOD_NAMES = frozenset({"mm", "bmm", "matmul", "addmm"})

_GATHER_FUNC_TARGETS = {
    aten.gather.default,
    aten.gather,
    torch.gather,
}

_METADATA_ONLY_FUNC_TARGETS = {
    aten.as_strided.default,
    aten.view.default,
    aten._unsafe_view.default,  # pylint: disable=protected-access
    aten.slice.Tensor,
    aten.select.int,
    aten.unsqueeze.default,
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.unsqueeze,
    aten.permute.default,
    aten.transpose.int,
    aten.t.default,
    aten.movedim.int,
    aten.movedim.intlist,
    aten.alias.default,
    aten.narrow.default,
}

_INDEXING_VIEW_FUNC_TARGETS = {
    aten.slice.Tensor,
    aten.select.int,
    aten.narrow.default,
}

# reshape / flatten may implicitly copy when the input is not view-compatible.
_MAY_COPY_FUNC_TARGETS = {
    aten.reshape.default,
    aten.reshape,
    aten.flatten.using_ints,
    torch.reshape,
    torch.flatten,
}

_VIEW_FUNC_TARGETS = _METADATA_ONLY_FUNC_TARGETS | _MAY_COPY_FUNC_TARGETS

_METADATA_ONLY_METHOD_NAMES = frozenset({
    "view",
    "expand",
    "unsqueeze",
    "squeeze",
    "permute",
    "transpose",
    "as_strided",
    "t",
    "movedim",
})

_MAY_COPY_METHOD_NAMES = frozenset({"reshape", "flatten"})

_VIEW_METHOD_NAMES = _METADATA_ONLY_METHOD_NAMES | _MAY_COPY_METHOD_NAMES

# Shared with C++ LayoutKind in view_utils.h. Python stamps this on rewritten views;
# C++ re-validates from packed as_strided size/stride/offset.
class LayoutKind:
    DENSE = "dense"
    BROADCAST = "broadcast"
    PERMUTE = "permute"
    GENERIC_STRIDED = "generic_strided"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LayoutDescriptor:
    """FX-side layout contract: storage root + ATen (size, stride, offset) + kind."""

    storage_root: Optional[Node]
    logical_shape: Optional[tuple]
    stride: Optional[tuple]
    storage_offset: Any
    kind: str
    permute_dims: Optional[tuple] = None


# Original aten.expand size, reused when inserting the materializing expand.
_LAYOUT_EXPAND_SIZE_META = "_fxrt_expand_size"
# Original permute dim order (python ints), reused when inserting Path C Transpose.
_LAYOUT_PERMUTE_DIMS_META = "_fxrt_permute_dims"
_LAYOUT_KIND_META = "_fxrt_layout_kind"


def _node_tensor_meta(node: Node) -> Optional[Any]:
    example = node.meta.get("example_value", None)
    if example is None:
        example = node.meta.get("val", None)
    return example


def _is_tensor_meta(value: Any) -> bool:
    return isinstance(value, (torch.Tensor, FakeTensor))


def _is_py_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same_int_or_sym(lhs: Any, rhs: Any) -> bool:
    """True for equal ints, identical SymInts, or equal symbolic expressions."""
    if _is_py_int(lhs) and _is_py_int(rhs):
        return lhs == rhs
    if isinstance(lhs, torch.SymInt) and isinstance(rhs, torch.SymInt):
        if lhs is rhs:
            return True
        try:
            return lhs.node.expr == rhs.node.expr
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
    # Constant SymInt (e.g. weight N) vs Python int. Never int(SymInt).
    if isinstance(lhs, torch.SymInt) and _is_py_int(rhs):
        try:
            return lhs.node.expr == rhs
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
    if isinstance(rhs, torch.SymInt) and _is_py_int(lhs):
        try:
            return rhs.node.expr == lhs
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
    return False


def _meta_storage_offset_from_example(example: Any) -> Optional[Any]:
    if not _is_tensor_meta(example):
        return None
    try:
        offset = example.storage_offset()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if _is_py_int(offset) or isinstance(offset, torch.SymInt):
        return offset
    return None


def _meta_storage_offset(node: Optional[Node]) -> Optional[Any]:
    """Return the real ATen storage offset without forcing a SymInt to int.

    make_fx FakeTensor metadata often reports offset 0 for slice/select views.
    Prefer a graph-inferred offset when one can be proved; otherwise use meta.
    """
    if node is None:
        return None
    inferred = _infer_view_storage_offset(node)
    if inferred is not None:
        return inferred
    return _meta_storage_offset_from_example(_node_tensor_meta(node))


def _expand_size_arg(node: Node) -> Optional[Any]:
    if node.op == "call_function" and len(node.args) >= 2:
        return node.args[1]
    if node.op == "call_method" and node.target == "expand" and len(node.args) >= 2:
        return list(node.args[1:])
    return None


def _sanitize_expand_size(size_arg: Any) -> Optional[Any]:
    """Keep FX nodes / Python ints; map SymInt to expand keep-dim (-1)."""
    if isinstance(size_arg, Node):
        return size_arg
    if not isinstance(size_arg, (list, tuple)):
        return None
    sanitized: list[Any] = []
    for dim in size_arg:
        if isinstance(dim, Node) or _is_py_int(dim):
            sanitized.append(dim)
        elif isinstance(dim, torch.SymInt):
            sanitized.append(-1)
        else:
            return None
    return sanitized


def _expand_size_from_meta(example: Any) -> list[int]:
    """Build aten.expand size without baking SymInt hints (e.g. 120)."""
    size: list[int] = []
    for dim in example.shape:
        size.append(dim if _is_py_int(dim) else -1)
    return size


def _lift_int_or_symint_list(graph: Any, values: list[Any]) -> list[Any]:
    """Turn size/stride into FX args. Python ints stay; SymInts are materialized."""
    if all(_is_py_int(v) for v in values):
        return list(values)
    try:
        return graph.materialize_symints(values)
    except (AssertionError, RuntimeError, TypeError, ValueError, AttributeError):
        # Do not int(SymInt). Raw SymInt may warn; -1 is invalid as_strided size.
        # AttributeError: torch >= 2.9 removed Graph.materialize_symints, and the sizes
        # are already plain ints/SymInts the caller can use as-is.
        return list(values)


def _is_expand_node(node: Node) -> bool:
    if node.op == "call_function" and node.target in _EXPAND_TARGETS:
        return True
    return node.op == "call_method" and node.target == "expand"


def _is_permute_node(node: Node) -> bool:
    if node.op == "call_function" and node.target in _PERMUTE_FUNC_TARGETS:
        return True
    return node.op == "call_method" and node.target in _PERMUTE_METHOD_NAMES


def _is_as_strided_node(node: Node) -> bool:
    if node.op == "call_function" and node.target in _AS_STRIDED_FUNC_TARGETS:
        return True
    return node.op == "call_method" and node.target == "as_strided"


def _as_strided_base(node: Node) -> Optional[Node]:
    if not _is_as_strided_node(node) or not node.args:
        return None
    base = node.args[0]
    return base if isinstance(base, Node) else None


def _normalized_dim(dim: Any, rank: Any) -> Optional[int]:
    if not _is_py_int(dim) or not _is_py_int(rank):
        return None
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    return dim


def _clamped_slice_start(start: Any, size: int, step: Any) -> Optional[int]:
    """Match ATen positive-step slice start clamping. None if not a Python int."""
    if not _is_py_int(start) or not _is_py_int(step) or step <= 0:
        return None
    if start < 0:
        start += size
    return max(0, min(start, size))


def _infer_index_storage_offset(
    base_offset: Any, base_example: Any, dim: int, index: int
) -> Optional[Any]:
    try:
        stride = base_example.stride(dim)
    except (RuntimeError, TypeError, ValueError, IndexError):
        return None
    return base_offset + index * stride


def _base_dim_extent(base_example: Any, dim_arg: Any) -> Optional[tuple]:
    """Return (normalized_dim, extent) when both are Python ints."""
    dim = _normalized_dim(dim_arg, base_example.dim())
    if dim is None:
        return None
    size = base_example.size(dim)
    if not _is_py_int(size):
        return None
    return dim, size


def _infer_slice_offset(node: Node, base_example: Any, base_offset: Any) -> Optional[Any]:
    """Storage offset for aten.slice, using ATen positive-step start clamping."""
    if len(node.args) < 3:
        return None
    extent = _base_dim_extent(base_example, node.args[1])
    if extent is None:
        return None
    dim, size = extent
    step = node.args[4] if len(node.args) >= 5 else 1
    start = _clamped_slice_start(node.args[2], size, step)
    if start is None:
        return None
    return _infer_index_storage_offset(base_offset, base_example, dim, start)


def _infer_narrow_offset(node: Node, base_example: Any, base_offset: Any) -> Optional[Any]:
    """Storage offset for aten.narrow when start and length are in-range Python ints."""
    if len(node.args) < 4 or not _is_py_int(node.args[2]) or not _is_py_int(node.args[3]):
        return None
    extent = _base_dim_extent(base_example, node.args[1])
    if extent is None:
        return None
    dim, size = extent
    start = node.args[2]
    if start < 0:
        start += size
    length = node.args[3]
    if start < 0 or start + length > size:
        return None
    return _infer_index_storage_offset(base_offset, base_example, dim, start)


def _infer_select_offset(node: Node, base_example: Any, base_offset: Any) -> Optional[Any]:
    """Storage offset for aten.select when the index is an in-range Python int."""
    if len(node.args) < 3 or not _is_py_int(node.args[2]):
        return None
    extent = _base_dim_extent(base_example, node.args[1])
    if extent is None:
        return None
    dim, size = extent
    index = node.args[2]
    if index < 0:
        index += size
    if index < 0 or index >= size:
        return None
    return _infer_index_storage_offset(base_offset, base_example, dim, index)


def _infer_slice_like_offset(node: Node, base: Node, base_offset: Any) -> Optional[Any]:
    """Prove storage offset for slice / narrow / select, else None."""
    base_example = _node_tensor_meta(base)
    if not _is_tensor_meta(base_example) or node.op != "call_function":
        return None
    target = node.target
    if target == aten.slice.Tensor:
        return _infer_slice_offset(node, base_example, base_offset)
    if target == aten.narrow.default:
        return _infer_narrow_offset(node, base_example, base_offset)
    if target == aten.select.int:
        return _infer_select_offset(node, base_example, base_offset)
    return None


def _infer_view_storage_offset(node: Node) -> Optional[Any]:
    """Prove a view's storage offset from the FX graph when FakeTensor drops it."""
    if _is_as_strided_node(node) and len(node.args) >= 4:
        offset = node.args[3]
        if _is_py_int(offset) or isinstance(offset, torch.SymInt):
            return offset
        if offset is not None:
            return None
    if not node.args or not isinstance(node.args[0], Node):
        return None
    base = node.args[0]
    base_offset = _meta_storage_offset(base)
    if base_offset is None:
        return None
    sliced = _infer_slice_like_offset(node, base, base_offset)
    if sliced is not None:
        return sliced
    if _is_expand_node(node) or _is_permute_node(node):
        return base_offset
    # reshape/flatten copy unless the input is still a contiguous alias.
    if _is_maybe_copying_view(node):
        return base_offset if _meta_is_contiguous(base) else 0
    if (
        _is_metadata_only_view(node)
        and not (node.op == "call_function" and node.target in _INDEXING_VIEW_FUNC_TARGETS)
        and not _is_as_strided_node(node)
    ):
        return base_offset
    return None


def _expand_prefix_axis_ok(out_s: Any) -> Optional[bool]:
    """True when a newly prepended axis is stride-0; None when invalid."""
    if not _same_int_or_sym(out_s, 0):
        return None
    return True


def _expand_mapped_axis_ok(out_dim: Any, out_s: Any, in_dim: Any, in_s: Any) -> Optional[bool]:
    """True if this aligned axis expands, False if retained, None if invalid."""
    if _same_int_or_sym(out_s, 0):
        singleton_input = _same_int_or_sym(in_dim, 1)
        existing_broadcast = _same_int_or_sym(out_dim, in_dim) and _same_int_or_sym(in_s, 0)
        if not singleton_input and not existing_broadcast:
            return None
        return not _same_int_or_sym(out_dim, 1)
    if not _same_int_or_sym(out_dim, in_dim) or not _same_int_or_sym(out_s, in_s):
        return None
    return False


def _meta_is_pure_expand_view(node: Node) -> bool:
    """Validate that a stride-zero view is exactly ATen expand geometry.

    A zero stride alone is insufficient: ``as_strided(x, (3, 2), (0, 2))``
    is an overlapping/generic view, not ``expand``. Input and output dimensions
    are right-aligned, every retained dimension keeps its input stride, and a
    zero stride may only expand a new or singleton dimension.
    """
    base = _as_strided_base(node)
    example = _node_tensor_meta(node)
    base_example = _node_tensor_meta(base) if base is not None else None
    if not _is_tensor_meta(example) or not _is_tensor_meta(base_example):
        return False
    out_shape = list(example.shape)
    out_stride = list(example.stride())
    in_shape = list(base_example.shape)
    in_stride = list(base_example.stride())
    if len(out_shape) < len(in_shape) or len(out_shape) != len(out_stride) or len(in_shape) != len(in_stride):
        return False
    offset = _meta_storage_offset(node)
    base_offset = _meta_storage_offset(base)
    if offset is None or base_offset is None or not _same_int_or_sym(offset, base_offset):
        return False
    rank_delta = len(out_shape) - len(in_shape)
    expanded = False
    for out_i, (out_dim, out_s) in enumerate(zip(out_shape, out_stride)):
        if out_i < rank_delta:
            axis_exp = _expand_prefix_axis_ok(out_s)
        else:
            in_i = out_i - rank_delta
            axis_exp = _expand_mapped_axis_ok(out_dim, out_s, in_shape[in_i], in_stride[in_i])
        if axis_exp is None:
            return False
        expanded = expanded or axis_exp
    return expanded


def _meta_has_broadcast_stride(node: Node) -> bool:
    """True only for a validated expand-style view, never generic as_strided."""
    return _layout_kind_of(node) == LayoutKind.BROADCAST


def _meta_has_real_broadcast_stride(node: Node) -> bool:
    """True when metadata has a stride-zero dimension with multiple elements."""
    example = _node_tensor_meta(node)
    if not _is_tensor_meta(example):
        return False
    return any(
        _same_int_or_sym(stride, 0) and not _same_int_or_sym(dim, 0) and not _same_int_or_sym(dim, 1)
        for dim, stride in zip(example.shape, example.stride())
    )


def _is_view_consumer(user: Node) -> bool:
    if user.op == "call_function" and user.target in _VIEW_FUNC_TARGETS:
        return True
    return user.op == "call_method" and user.target in _VIEW_METHOD_NAMES


def _is_metadata_only_view(user: Node) -> bool:
    """True for view ops that never implicitly copy (unlike reshape/flatten)."""
    if user.op == "call_function" and user.target in _METADATA_ONLY_FUNC_TARGETS:
        return True
    return user.op == "call_method" and user.target in _METADATA_ONLY_METHOD_NAMES


def _is_maybe_copying_view(user: Node) -> bool:
    """True for reshape/flatten, which may implicitly copy a non-contiguous input."""
    if user.op == "call_function" and user.target in _MAY_COPY_FUNC_TARGETS:
        return True
    return user.op == "call_method" and user.target in _MAY_COPY_METHOD_NAMES


def _is_compiled_kernel_user(user: Node) -> bool:
    target = user.target
    if getattr(target, "__name__", None) == "compiled_kernel_wrapper_mutation":
        return True
    return str(target) == "torch.ops.higher_order.compiled_kernel_wrapper_mutation"


def _is_bmm_node(node: Node) -> bool:
    if node.op == "call_function" and node.target in _BMM_FUNC_TARGETS:
        return True
    return node.op == "call_method" and node.target == "bmm"


def _is_mm_family_user(user: Node) -> bool:
    if user.op == "call_function" and user.target in _MM_FAMILY_FUNC_TARGETS:
        return True
    return user.op == "call_method" and user.target in _MM_FAMILY_METHOD_NAMES


def _is_last_two_perm(perm: Optional[list[int]]) -> bool:
    if not isinstance(perm, list) or len(perm) < 2:
        return False
    rank = len(perm)
    return perm[: rank - 2] == list(range(rank - 2)) and perm[rank - 2 :] == [rank - 1, rank - 2]


def _is_gather_broadcast_index(user: Node, producer: Optional[Node]) -> bool:
    """Return whether producer is the index operand of aten.gather."""
    if producer is None or len(user.args) < 3:
        return False
    is_gather = (
        user.op == "call_function" and user.target in _GATHER_FUNC_TARGETS
    ) or (user.op == "call_method" and user.target == "gather")
    return is_gather and user.args[2] is producer


def _gather_broadcast_rank_compatible(producer: Optional[Node]) -> bool:
    """C++ Gather descriptor supports broadcast only when physical/logical ranks match."""
    if producer is None:
        return False
    base = _as_strided_base(producer)
    example = _node_tensor_meta(producer)
    base_example = _node_tensor_meta(base) if base is not None else None
    return _is_tensor_meta(example) and _is_tensor_meta(base_example) and example.dim() == base_example.dim()


def _meta_is_contiguous(node: Optional[Node]) -> bool:
    if node is None:
        return False
    example = _node_tensor_meta(node)
    if not _is_tensor_meta(example):
        return False
    try:
        return bool(example.is_contiguous())
    except (RuntimeError, TypeError, ValueError):
        return False


def _user_lowers_to_fallback(user: Node) -> bool:
    """True when this user becomes custom_call or python_call.

    Those kernels are rebuilt from the view stride, so the edge must stay a
    view. Catalog ops stay on the dense edge.
    """
    if user.op not in ("call_function", "call_method"):
        return False
    # Local import: legalize runs after op_support.register, and importing the
    # rewrite package at module load would cycle through the graph builder.
    from fxrt.ir import Op  # pylint: disable=import-outside-toplevel
    from fxrt import fx_backend as fxb  # pylint: disable=import-outside-toplevel
    from fxrt.gert_backend.rewrites.op_support import (  # pylint: disable=import-outside-toplevel
        _converter_supported_logical_ops,
        _forced_custom_call_names,
    )

    target = user.target
    if user.op == "call_function":
        op = fxb._get_op(target)  # pylint: disable=protected-access
    else:
        op = fxb._OP_MAP.get(target)  # pylint: disable=protected-access
    if op is None or op == Op.custom_call:
        return user.op == "call_function" and callable(target)
    if op.name in _forced_custom_call_names():
        return True
    return op.name not in _converter_supported_logical_ops()


def _broadcast_consumer_needs_dense(user: Node, producer: Node):
    """Whether a broadcast producer must be materialized.

    ``None`` means the producer is not a broadcast. Expand is a metadata view,
    but unsqueeze/slice/alias consumers do not all preserve stride-zero
    semantics, so those edges are materialized.
    """
    desc = _layout_descriptor(producer)
    if desc is None or desc.kind != LayoutKind.BROADCAST:
        return None
    if _is_as_strided_node(user) or _is_permute_node(user):
        return False
    if _user_lowers_to_fallback(user):
        return False
    return True


def _permute_view_needs_dense(user: Node, producer: Optional[Node], perm: Optional[list[int]]) -> bool:
    """Whether a permute view must be materialized for this consumer.

    A swap involving a size-1 dimension can still be contiguous, so a following
    reshape/view can alias it. DualStage carries the last-two permutation as
    transpose_x1/x2 and keeps the original storage address.
    """
    if _is_as_strided_node(user) or _is_permute_node(user):
        return False
    if _is_metadata_only_view(user) and _meta_is_contiguous(producer):
        return False
    if _is_mm_family_user(user) and _is_last_two_perm(perm):
        return False
    if _user_lowers_to_fallback(user):
        return False
    return True


def _consumer_needs_dense(
    user: Node,
    producer: Optional[Node] = None,
    permute_view: bool = False,
    perm: Optional[list[int]] = None,
) -> bool:
    """True when a consumer needs a dense (materialized) input from its producer view.

    Output consumers and catalog compute users need dense. compiled kernels,
    fallback (custom_call / python_call), and follow-up views keep the alias.
    """
    if user.op == "output":
        return True
    if _is_compiled_kernel_user(user):
        return False
    if _is_gather_broadcast_index(user, producer):
        return not _gather_broadcast_rank_compatible(producer)
    if _is_maybe_copying_view(user):
        return not _meta_is_contiguous(producer)
    if producer is not None:
        broadcast = _broadcast_consumer_needs_dense(user, producer)
        if broadcast is not None:
            return broadcast
    if permute_view:
        return _permute_view_needs_dense(user, producer, perm)
    if _is_view_consumer(user):
        return False
    # custom_call / python_call rebuild the tensor from its stride. Catalog
    # users of the same view still take the dense edge above.
    if _user_lowers_to_fallback(user):
        return False
    return True


def _copy_meta(dst: Node, src: Node) -> None:
    dst.meta.update(src.meta)


def _set_contiguous_meta(dst: Node, example: Any) -> None:
    if not _is_tensor_meta(example):
        return
    try:
        dense = example.contiguous()
    except (RuntimeError, ValueError, TypeError):
        return
    dst.meta["example_value"] = dense
    dst.meta["val"] = dense


def _set_dense_meta(dst: Node, src: Node) -> None:
    _copy_meta(dst, src)
    _set_contiguous_meta(dst, _node_tensor_meta(src))


def _normalize_dim(dim: Any, rank: int) -> Optional[int]:
    if not _is_py_int(dim):
        return None
    resolved = dim + rank if dim < 0 else dim
    if resolved < 0 or resolved >= rank:
        return None
    return resolved


def _dims_list_from_arg(arg: Any, rank: int) -> Optional[list[int]]:
    """Normalize a permute dims argument (int / list / tuple) into a full permutation list."""
    if isinstance(arg, Node) or rank < 0:
        return None
    values = arg
    if _is_py_int(arg):
        values = (arg,)
    elif not isinstance(arg, (list, tuple)):
        return None
    out: list[int] = []
    for dim in values:
        resolved = _normalize_dim(dim, rank)
        if resolved is None:
            return None
        out.append(resolved)
    if len(out) != rank or sorted(out) != list(range(rank)):
        return None
    return out


def _identity_perm(rank: int) -> list[int]:
    return list(range(rank))


def _t_perm(rank: int) -> Optional[list[int]]:
    if rank < 0:
        return None
    if rank < 2:
        return _identity_perm(rank)
    return list(range(rank - 2)) + [rank - 1, rank - 2]


def _transpose_perm(dim0: Any, dim1: Any, rank: int) -> Optional[list[int]]:
    resolved0 = _normalize_dim(dim0, rank)
    resolved1 = _normalize_dim(dim1, rank)
    if resolved0 is None or resolved1 is None:
        return None
    perm = _identity_perm(rank)
    perm[resolved0], perm[resolved1] = perm[resolved1], perm[resolved0]
    return perm


def _movedim_perm(source: Any, destination: Any, rank: int) -> Optional[list[int]]:
    """Permutation implied by movedim(source, destination) on a rank-dim tensor."""
    def _as_dims(arg: Any) -> Optional[list[int]]:
        if _is_py_int(arg):
            resolved = _normalize_dim(arg, rank)
            return None if resolved is None else [resolved]
        if not isinstance(arg, (list, tuple)):
            return None
        out: list[int] = []
        for dim in arg:
            resolved = _normalize_dim(dim, rank)
            if resolved is None:
                return None
            out.append(resolved)
        return out

    src = _as_dims(source)
    dst = _as_dims(destination)
    if src is None or dst is None or len(src) != len(dst):
        return None
    if len(set(src)) != len(src) or len(set(dst)) != len(dst):
        return None
    rest = [idx for idx in range(rank) if idx not in src]
    dest_to_src = {dst[i]: src[i] for i in range(len(src))}
    perm: list[int] = []
    rest_i = 0
    for idx in range(rank):
        if idx in dest_to_src:
            perm.append(dest_to_src[idx])
        else:
            perm.append(rest[rest_i])
            rest_i += 1
    return perm


def _permute_dims_from_function(node: Node, rank: int) -> Optional[list[int]]:
    """Permutation of a functional t / transpose / permute / movedim call, None when unreadable."""
    target = node.target
    if target in (aten.t.default, torch.t):
        return _t_perm(rank)
    if target in (aten.transpose.int, torch.transpose) and len(node.args) >= 3:
        return _transpose_perm(node.args[1], node.args[2], rank)
    if target in (aten.permute.default, aten.permute, torch.permute) and len(node.args) >= 2:
        return _dims_list_from_arg(node.args[1], rank)
    if target in (aten.movedim.int, aten.movedim.intlist, torch.movedim) and len(node.args) >= 3:
        return _movedim_perm(node.args[1], node.args[2], rank)
    return None


def _permute_dims_from_method(node: Node, rank: int) -> Optional[list[int]]:
    """Permutation of a tensor-method t / transpose / permute / movedim call, None when unreadable."""
    if node.target == "t":
        return _t_perm(rank)
    if node.target == "transpose" and len(node.args) >= 3:
        return _transpose_perm(node.args[1], node.args[2], rank)
    if node.target == "permute" and len(node.args) >= 2:
        rest = list(node.args[1:])
        if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
            return _dims_list_from_arg(rest[0], rank)
        return _dims_list_from_arg(rest, rank)
    if node.target == "movedim" and len(node.args) >= 3:
        return _movedim_perm(node.args[1], node.args[2], rank)
    return None


def _permute_dims_from_node(node: Node, rank: int) -> Optional[list[int]]:
    """Permutation of a t / transpose / permute / movedim node, None when unreadable."""
    if rank < 0:
        return None
    if node.op == "call_function":
        return _permute_dims_from_function(node, rank)
    if node.op == "call_method":
        return _permute_dims_from_method(node, rank)
    return None


def _compose_perm(first: list[int], second: list[int]) -> Optional[list[int]]:
    if len(first) != len(second):
        return None
    try:
        return [first[idx] for idx in second]
    except (IndexError, TypeError):
        return None


def _infer_perm_from_meta(base_ex: Any, view_ex: Any) -> Optional[list[int]]:
    """Infer a permutation by matching size+stride pairs between base and view examples."""
    in_shape = list(base_ex.shape)
    in_stride = list(base_ex.stride())
    out_shape = list(view_ex.shape)
    out_stride = list(view_ex.stride())
    if len(in_shape) != len(out_shape) or len(out_shape) != len(out_stride):
        return None
    used = [False] * len(in_shape)
    perm: list[int] = []
    for out_size, out_stride_i in zip(out_shape, out_stride):
        found: Optional[int] = None
        for idx, (in_size, in_stride_i) in enumerate(zip(in_shape, in_stride)):
            if used[idx]:
                continue
            if _same_int_or_sym(out_size, in_size) and _same_int_or_sym(out_stride_i, in_stride_i):
                found = idx
                break
        if found is None:
            return None
        used[found] = True
        perm.append(found)
    if perm == _identity_perm(len(perm)):
        return None
    return perm


def _infer_perm_ignoring_broadcast(base_ex: Any, view_ex: Any) -> Optional[list[int]]:
    """Match non-broadcast view dims to base dims by size+stride.

    Inductor emits a single as_strided for ``w.t().expand(B, N, N)``:
    base ``[N,N]/(N,1)``, view ``[B,N,N]/(0,1,N)``. The stride-0 dims are
    expand; the rest must be a full permutation of the base axes.
    """
    in_shape = list(base_ex.shape)
    in_stride = list(base_ex.stride())
    out_shape = list(view_ex.shape)
    out_stride = list(view_ex.stride())
    if not in_shape or len(out_shape) != len(out_stride):
        return None
    used = [False] * len(in_shape)
    perm: list[int] = []
    for out_size, out_stride_i in zip(out_shape, out_stride):
        if _is_py_int(out_stride_i) and out_stride_i == 0:
            continue
        found: Optional[int] = None
        for idx, (in_size, in_stride_i) in enumerate(zip(in_shape, in_stride)):
            if used[idx]:
                continue
            if _same_int_or_sym(out_size, in_size) and _same_int_or_sym(out_stride_i, in_stride_i):
                found = idx
                break
        if found is None:
            return None
        used[found] = True
        perm.append(found)
    if len(perm) != len(in_shape) or sorted(perm) != list(range(len(in_shape))):
        return None
    if perm == _identity_perm(len(perm)):
        return None
    return perm


def _permute_dims_for_view(node: Node) -> Optional[list[int]]:
    """Permutation a view node applies over its storage, from stored meta or shape inference."""
    stored = node.meta.get(_LAYOUT_PERMUTE_DIMS_META)
    if isinstance(stored, list) and stored and all(_is_py_int(dim) for dim in stored):
        return list(stored)
    base = _same_rank_storage_root(node)
    if base is None:
        return None
    example = _node_tensor_meta(node)
    base_ex = _node_tensor_meta(base)
    if not _is_tensor_meta(example) or not _is_tensor_meta(base_ex):
        return None
    if _meta_has_real_broadcast_stride(node):
        return None
    perm = _infer_perm_from_meta(base_ex, example)
    if perm is not None:
        return perm
    if _looks_like_rank2_transpose(base_ex, example):
        return [1, 0]
    return None


def _layout_kind_of(node: Node) -> str:
    """Infer LayoutKind; prefer the kind stamped when expand/permute were rewritten."""
    stored = node.meta.get(_LAYOUT_KIND_META)
    if stored in (LayoutKind.DENSE, LayoutKind.BROADCAST, LayoutKind.PERMUTE, LayoutKind.GENERIC_STRIDED):
        return stored
    if _meta_is_pure_expand_view(node):
        return LayoutKind.BROADCAST
    if _meta_has_real_broadcast_stride(node):
        return LayoutKind.GENERIC_STRIDED
    if _permute_dims_for_view(node) is not None:
        return LayoutKind.PERMUTE
    if _meta_is_contiguous(node):
        return LayoutKind.DENSE
    if _is_as_strided_node(node):
        return LayoutKind.GENERIC_STRIDED
    return LayoutKind.UNKNOWN


def _layout_descriptor(node: Optional[Node]) -> Optional[LayoutDescriptor]:
    """Build the FX layout descriptor. C++ re-validates the same kind from packed layout."""
    if node is None:
        return None
    example = _node_tensor_meta(node)
    perm = _permute_dims_for_view(node)
    return LayoutDescriptor(
        storage_root=_same_rank_storage_root(node),
        logical_shape=tuple(example.shape) if _is_tensor_meta(example) else None,
        stride=tuple(example.stride()) if _is_tensor_meta(example) else None,
        storage_offset=_meta_storage_offset(node),
        kind=_layout_kind_of(node),
        permute_dims=tuple(perm) if perm is not None else None,
    )


def fold_consecutive_permutes_(gm: GraphModule) -> None:
    """Compose permute/transpose/t chains into one permute of the original tensor."""
    graph = gm.graph
    changed_any = False
    n_nodes = len(list(graph.nodes))
    for _ in range(max(n_nodes, 1)):
        changed = False
        for node in list(graph.nodes):
            if not _is_permute_node(node) or not node.args:
                continue
            inner = node.args[0]
            if not isinstance(inner, Node) or not _is_permute_node(inner) or not inner.args:
                continue
            example = _node_tensor_meta(node)
            if not _is_tensor_meta(example):
                continue
            rank = example.dim()
            if not _is_py_int(rank):
                continue
            first = _permute_dims_from_node(inner, rank)
            second = _permute_dims_from_node(node, rank)
            if first is None or second is None:
                continue
            composed = _compose_perm(first, second)
            inner_base = inner.args[0]
            if composed is None or not isinstance(inner_base, Node):
                continue
            with graph.inserting_before(node):
                folded = graph.call_function(aten.permute.default, (inner_base, composed))
            _copy_meta(folded, node)
            node.replace_all_uses_with(folded)
            graph.erase_node(node)
            if not inner.users:
                graph.erase_node(inner)
            changed = True
            changed_any = True
        if not changed:
            break
    if changed_any:
        graph.lint()
        gm.recompile()


def rewrite_expand_as_view_(gm: GraphModule) -> None:
    """Replace aten.expand with as_strided using Torch expand geometry (from meta)."""
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if not _is_expand_node(node) or not node.args:
            continue
        base = node.args[0]
        if not isinstance(base, Node):
            continue
        example = _node_tensor_meta(node)
        if not _is_tensor_meta(example):
            continue
        orig_size = _sanitize_expand_size(_expand_size_arg(node))
        size_vals = list(example.shape)
        stride_vals = list(example.stride())
        offset = _meta_storage_offset(node)
        if offset is None:
            continue
        with graph.inserting_before(node):
            lifted = _lift_int_or_symint_list(graph, size_vals + stride_vals)
            n_size = len(size_vals)
            lifted_offset = _lift_int_or_symint_list(graph, [offset])[0]
            view = graph.call_function(
                aten.as_strided.default, (base, lifted[:n_size], lifted[n_size:], lifted_offset)
            )
        _copy_meta(view, node)
        view.meta[_LAYOUT_KIND_META] = LayoutKind.BROADCAST
        if orig_size is not None:
            view.meta[_LAYOUT_EXPAND_SIZE_META] = orig_size
        node.replace_all_uses_with(view)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()


def rewrite_permute_as_view_(gm: GraphModule) -> None:
    """Replace permute/transpose/t with as_strided using Torch permute geometry."""
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if not _is_permute_node(node) or not node.args:
            continue
        base = node.args[0]
        if not isinstance(base, Node):
            continue
        example = _node_tensor_meta(node)
        if not _is_tensor_meta(example):
            continue
        rank = example.dim()
        if not _is_py_int(rank):
            continue
        perm = _permute_dims_from_node(node, rank)
        if perm is None or perm == _identity_perm(rank):
            continue
        size_vals = list(example.shape)
        stride_vals = list(example.stride())
        offset = _meta_storage_offset(node)
        if offset is None:
            continue
        with graph.inserting_before(node):
            lifted = _lift_int_or_symint_list(graph, size_vals + stride_vals)
            n_size = len(size_vals)
            lifted_offset = _lift_int_or_symint_list(graph, [offset])[0]
            view = graph.call_function(
                aten.as_strided.default, (base, lifted[:n_size], lifted[n_size:], lifted_offset)
            )
        _copy_meta(view, node)
        view.meta[_LAYOUT_KIND_META] = LayoutKind.PERMUTE
        view.meta[_LAYOUT_PERMUTE_DIMS_META] = perm
        node.replace_all_uses_with(view)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()


def _has_compiled_kernel_user(node: Node) -> bool:
    return any(_is_compiled_kernel_user(user) for user in node.users)


def _legalize_broadcast_node(graph: Any, node: Node) -> bool:
    """Materialize an expand view for consumers that need dense input; True when rewritten."""
    example = _node_tensor_meta(node)
    if not _is_tensor_meta(example):
        return False
    # compiled_kernel_wrapper_mutation writes the view in place and FX→IR
    # rebinds env[view]=kernel. Materializing a sibling softmax/expand from
    # the pre-mutation base leaves softmax on the stale -1e9 pad.
    if _has_compiled_kernel_user(node):
        return False
    dense_users = [user for user in list(node.users) if _consumer_needs_dense(user, producer=node)]
    if not dense_users:
        return False
    size = node.meta.get(_LAYOUT_EXPAND_SIZE_META)
    if size is None:
        size = _expand_size_from_meta(example)
    # Expand the unexpanded tensor, not the already-logical view. Path C
    # BroadcastTo compares data StorageShape to the target; feeding the
    # as_strided view (whose ViewAlias physical shape is the storage root)
    # turns unsqueeze([B,8,1]).expand(..., 128) into expand([B,8] → [B,8,128])
    # and Execute fails: [128] cannot broadcast to [8].
    base = _as_strided_base(node)
    src = base if base is not None else node
    with graph.inserting_after(node):
        materialized = graph.call_function(aten.expand.default, (src, size))
    _set_dense_meta(materialized, node)
    for user in dense_users:
        user.replace_input_with(node, materialized)
    return True


def _legalize_permute_node(graph: Any, node: Node, perm: list[int]) -> bool:
    """Materialize permute(base, perm) for consumers that need dense input; True when rewritten."""
    # Insert permute(base, perm), not permute(view): the view is already logical.
    if _has_compiled_kernel_user(node):
        return False
    dense_users = [
        user for user in list(node.users)
        if _consumer_needs_dense(user, producer=node, permute_view=True, perm=perm)
    ]
    if not dense_users:
        return False
    base = _same_rank_storage_root(node)
    if base is None:
        return False
    with graph.inserting_after(node):
        materialized = graph.call_function(aten.permute.default, (base, list(perm)))
    _set_dense_meta(materialized, node)
    for user in dense_users:
        user.replace_input_with(node, materialized)
    return True


def _chain_permute_dims(node: Node) -> Optional[list[int]]:
    """Permutation of an as_strided view, following the chain through its base view."""
    perm = _permute_dims_for_view(node)
    if perm is not None:
        return perm
    base = _as_strided_base(node)
    if base is None:
        return None
    stored = base.meta.get(_LAYOUT_PERMUTE_DIMS_META)
    if isinstance(stored, list) and stored and all(_is_py_int(dim) for dim in stored):
        return list(stored)
    example = _node_tensor_meta(node)
    base_ex = _node_tensor_meta(base)
    if _is_tensor_meta(example) and _is_tensor_meta(base_ex):
        return _infer_perm_ignoring_broadcast(base_ex, example)
    return None


def _combo_root_and_metas(node: Node) -> tuple[Optional[Node], Any, Any, Any]:
    """Root tensor plus expand/permute metas for a combo as_strided view.

    Rewrite combo is as_strided(as_strided(root)). Inductor-native combo is a
    single as_strided(root, size, stride) — root is the as_strided input.
    """
    base = _as_strided_base(node)
    if base is None:
        return None, None, None, None
    if _is_as_strided_node(base) and base.args and isinstance(base.args[0], Node):
        return (
            base.args[0],
            base.meta.get(_LAYOUT_EXPAND_SIZE_META),
            node.meta.get(_LAYOUT_EXPAND_SIZE_META),
            base.meta.get(_LAYOUT_PERMUTE_DIMS_META),
        )
    return base, None, node.meta.get(_LAYOUT_EXPAND_SIZE_META), None


def _is_impure_layout_node(node: Node) -> bool:
    # compiled_kernel_wrapper_mutation has num_users=0 (in-place). Default DCE
    # would drop every Autofuse kernel in the ETA host graph.
    return _is_compiled_kernel_user(node) or node.is_impure()


def _inverse_perm(perm: list[int]) -> Optional[list[int]]:
    inv = [0] * len(perm)
    try:
        for dst, src in enumerate(perm):
            inv[src] = dst
    except (IndexError, TypeError):
        return None
    return inv


def _insert_materialize_pair(
    graph: Any, after: Node, first_target: Any, first_args: tuple, second_target: Any, second_args_tail: tuple
) -> tuple[Node, Node]:
    """Insert op pair in topological order: first after `after`, second after first."""
    with graph.inserting_after(after):
        first = graph.call_function(first_target, first_args)
    with graph.inserting_after(first):
        second = graph.call_function(second_target, (first,) + second_args_tail)
    return first, second


def _materialize_expand_then_permute(
    graph: Any, node: Node, root: Node, expand_on_base: Any, perm: list[int], example: Any
) -> Node:
    """Insert expand(root, expand_on_base) followed by permute; returns the dense result."""
    expanded, materialized = _insert_materialize_pair(
        graph, node, aten.expand.default, (root, expand_on_base), aten.permute.default, (list(perm),)
    )
    inv = _inverse_perm(perm)
    if inv is not None and _is_tensor_meta(example):
        try:
            _set_contiguous_meta(expanded, example.permute(*inv))
        except (RuntimeError, ValueError, TypeError, IndexError):
            _copy_meta(expanded, node)
    return materialized


def _materialize_permute_then_expand(
    graph: Any, node: Node, root: Node, apply_perm: list[int], expand_on_node: Any
) -> Node:
    """Insert permute(root, apply_perm) followed by expand; returns the dense result."""
    permuted, materialized = _insert_materialize_pair(
        graph, node, aten.permute.default, (root, list(apply_perm)), aten.expand.default, (expand_on_node,)
    )
    root_ex = _node_tensor_meta(root)
    if _is_tensor_meta(root_ex):
        try:
            _set_contiguous_meta(permuted, root_ex.permute(*apply_perm))
        except (RuntimeError, ValueError, TypeError, IndexError):
            _copy_meta(permuted, node)
    return materialized


def _legalize_combo_node(graph: Any, node: Node, perm: list[int]) -> bool:
    """Materialize expand+permute (or permute+expand) from the original tensor.

    Combo views are not a last-two swap of contiguous storage, so DualStage mm/bmm
    must not skip this materialization.
    """
    if _has_compiled_kernel_user(node):
        return False
    dense_users = [user for user in list(node.users) if _consumer_needs_dense(user, producer=node)]
    if not dense_users:
        return False
    root, expand_on_base, expand_on_node, inner_perm = _combo_root_and_metas(node)
    if root is None:
        return False
    example = _node_tensor_meta(node)
    if expand_on_node is None and _is_as_strided_node(node) and len(node.args) >= 2:
        # Inductor-native combo: reuse as_strided size (FX nodes / SymInts).
        # _expand_size_from_meta would emit [-1,-1,-1] and drop the new batch dim.
        expand_on_node = node.args[1]
    if expand_on_node is None and _is_tensor_meta(example):
        expand_on_node = _expand_size_from_meta(example)
    apply_perm = inner_perm if isinstance(inner_perm, list) else perm
    if expand_on_base is not None:
        materialized = _materialize_expand_then_permute(graph, node, root, expand_on_base, perm, example)
    elif expand_on_node is not None and apply_perm:
        materialized = _materialize_permute_then_expand(graph, node, root, apply_perm, expand_on_node)
    else:
        return False
    _set_dense_meta(materialized, node)
    for user in dense_users:
        user.replace_input_with(node, materialized)
    return True


def legalize_dense_uses_(gm: GraphModule) -> None:
    """Insert a materializing expand/permute on view edges whose consumer needs dense."""
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if _meta_has_real_broadcast_stride(node):
            combo_perm = _chain_permute_dims(node)
            if combo_perm is not None:
                if _legalize_combo_node(graph, node, combo_perm):
                    changed = True
                    continue
                # Combo inferred but insert failed: do not fall through to
                # expand-only (that drops the permute and silently wrongs bmm).
                continue
            if _meta_has_broadcast_stride(node) and _legalize_broadcast_node(graph, node):
                changed = True
            continue
        perm = _permute_dims_for_view(node)
        if perm is None:
            continue
        if _legalize_permute_node(graph, node, perm):
            changed = True
    if changed:
        graph.eliminate_dead_code(_is_impure_layout_node)
        graph.lint()
        gm.recompile()


def _is_batch_broadcast_rank3(operand: Node) -> bool:
    example = _node_tensor_meta(operand)
    if not _is_tensor_meta(example) or example.dim() != 3:
        return False
    batch_stride = example.stride(0)
    return _is_py_int(batch_stride) and batch_stride == 0


def _walk_as_strided_to_rank2(node: Node) -> Optional[Node]:
    """Walk an as_strided chain up to the rank-2 base tensor, None when it never reaches one."""
    cur: Optional[Node] = node
    seen: set[int] = set()
    while isinstance(cur, Node) and id(cur) not in seen:
        seen.add(id(cur))
        example = _node_tensor_meta(cur)
        if _is_tensor_meta(example) and example.dim() == 2:
            return cur
        if not _is_as_strided_node(cur) or not cur.args or not isinstance(cur.args[0], Node):
            return None
        cur = cur.args[0]
    return None


def _same_rank_storage_root(node: Node) -> Optional[Node]:
    """Skip same-rank as_strided views (t / permute) down to the storage tensor.

    C++ analogue: StorageRootSameRankOf. Inductor rewrite emits
    as_strided(t(w), [N,N], [1,N]) — identity of the already-transposed view.
    Legalize must permute the original weight, not t(w).
    """
    example = _node_tensor_meta(node)
    if not _is_tensor_meta(example):
        return _as_strided_base(node)
    rank = example.dim()
    cur: Optional[Node] = node
    last: Optional[Node] = None
    seen: set[int] = set()
    while isinstance(cur, Node) and id(cur) not in seen:
        seen.add(id(cur))
        cur_ex = _node_tensor_meta(cur)
        if _is_tensor_meta(cur_ex) and cur_ex.dim() == rank:
            last = cur
        if not _is_as_strided_node(cur):
            break
        parent = _as_strided_base(cur)
        if parent is None:
            break
        cur = parent
    if last is None or last is node:
        return _as_strided_base(node)
    return last


def _looks_like_rank2_transpose(base_ex: Any, view_ex: Any) -> bool:
    """True for a rank-2 t() view, including hinted N vs symbolic N.

    Inductor host graphs keep W as [s15,s53]/(s53,1) and emit
    as_strided(..., [128,128], [1,128]). Size equality would plant a guard
    (int(SymInt)), so match the contiguous→column-major stride pattern.
    """
    if not _is_tensor_meta(base_ex) or not _is_tensor_meta(view_ex):
        return False
    if base_ex.dim() != 2 or view_ex.dim() != 2:
        return False
    if _infer_perm_from_meta(base_ex, view_ex) == [1, 0]:
        return True
    base_stride = list(base_ex.stride())
    view_stride = list(view_ex.stride())
    if not (
        _is_py_int(base_stride[1])
        and base_stride[1] == 1
        and _is_py_int(view_stride[0])
        and view_stride[0] == 1
    ):
        return False
    if _same_int_or_sym(base_stride[0], view_stride[1]):
        return True
    if not (_is_py_int(view_stride[1]) and view_stride[1] > 1):
        return False
    out0, out1 = view_ex.size(0), view_ex.size(1)
    hinted_square = _is_py_int(out0) and _is_py_int(out1) and out0 == out1 == view_stride[1]
    in0, in1 = base_ex.size(0), base_ex.size(1)
    symbolic_square = isinstance(in0, torch.SymInt) and isinstance(in1, torch.SymInt)
    return hinted_square and (symbolic_square or _same_int_or_sym(in0, in1))


def _matches_rank2_layout(root_ex: Any, want_size: list, want_stride: list, want_offset: Any) -> bool:
    """True when the root example already carries the requested 2D size and stride."""
    return (
        all(_same_int_or_sym(lhs, rhs) for lhs, rhs in zip(root_ex.shape, want_size))
        and all(_same_int_or_sym(lhs, rhs) for lhs, rhs in zip(root_ex.stride(), want_stride))
        and _same_int_or_sym(_meta_storage_offset_from_example(root_ex), want_offset)
    )


def _rank2_view_example(root_ex: Any, want_size: list, want_stride: list, want_offset: Any) -> Any:
    """Example tensor of the 2D as_strided view; transpose fallback for hinted shapes."""
    try:
        return root_ex.as_strided(want_size, want_stride, want_offset)
    except (RuntimeError, ValueError, TypeError, IndexError):
        pass
    # Hinted [128,128] vs symbolic W: as_strided may refuse. t() is the ETA case.
    if _is_py_int(want_stride[0]) and want_stride[0] == 1 and _same_int_or_sym(want_offset, 0):
        try:
            return root_ex.transpose(0, 1)
        except (RuntimeError, ValueError, TypeError, IndexError):
            return None
    return None


def _resolve_batch_broadcast_matrix(graph: Any, operand: Node, before: Node) -> Optional[Node]:
    """2D matrix that bmm sees after dropping a stride-0 batch dim.

    ETA: ``as_strided(W, [B,N,N], [0,1,N])`` or ``W.t().expand(B,N,N)``. The
    last two size/stride are a rank-2 view of the root (often ``W.t()``).
    """
    example = _node_tensor_meta(operand)
    if not _is_tensor_meta(example) or example.dim() != 3:
        return None
    batch_stride = example.stride(0)
    if not (_is_py_int(batch_stride) and batch_stride == 0):
        return None
    rank2 = _walk_as_strided_to_rank2(operand)
    if rank2 is None:
        return None
    root = _same_rank_storage_root(rank2)
    if root is None:
        root = rank2
    root_ex = _node_tensor_meta(root)
    if not _is_tensor_meta(root_ex) or root_ex.dim() != 2:
        return None
    want_size = [example.size(1), example.size(2)]
    want_stride = [example.stride(1), example.stride(2)]
    want_offset = _meta_storage_offset(operand)
    if want_offset is None:
        return None
    if _matches_rank2_layout(root_ex, want_size, want_stride, want_offset):
        return root
    with graph.inserting_before(before):
        lifted = _lift_int_or_symint_list(graph, want_size + want_stride + [want_offset])
        view2d = graph.call_function(aten.as_strided.default, (root, lifted[:2], lifted[2:4], lifted[4]))
    view_ex = _rank2_view_example(root_ex, want_size, want_stride, want_offset)
    if view_ex is not None:
        view2d.meta["val"] = view_ex
        view2d.meta["example_value"] = view_ex
    perm = _infer_perm_from_meta(root_ex, _node_tensor_meta(view2d))
    if perm is None and _looks_like_rank2_transpose(root_ex, _node_tensor_meta(view2d)):
        perm = [1, 0]
    if perm is None and _is_py_int(want_stride[0]) and want_stride[0] == 1:
        perm = [1, 0]
    if perm is not None:
        view2d.meta[_LAYOUT_PERMUTE_DIMS_META] = perm
    return view2d


def rewrite_broadcast_bmm_to_mm_(gm: GraphModule) -> None:
    """bmm(x, W.expand(B, K, N)) → reshape + mm + reshape when W is rank-2.

    Avoids materializing [B,K,N] (Path C Expand) for ETA ``w.t().expand(B,N,N)``.
    The remaining 2D ``t()`` is Path C Transpose in front of mm.
    """
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if not _is_bmm_node(node) or len(node.args) < 2:
            continue
        lhs, rhs = node.args[0], node.args[1]
        if not isinstance(lhs, Node) or not isinstance(rhs, Node):
            continue
        # Right-hand broadcast only: out[b] = x[b] @ W  ⇒  mm(x.reshape(-1, K), W).
        # Left-hand broadcast needs a different flatten and is not in ETA.
        if _is_batch_broadcast_rank3(lhs) or not _is_batch_broadcast_rank3(rhs):
            continue
        x = lhs
        x_ex = _node_tensor_meta(x)
        bmm_ex = _node_tensor_meta(node)
        if not _is_tensor_meta(x_ex) or x_ex.dim() != 3:
            continue
        w2d = _resolve_batch_broadcast_matrix(graph, rhs, node)
        w_ex = _node_tensor_meta(w2d) if w2d is not None else None
        if w2d is None or not _is_tensor_meta(w_ex):
            continue
        k = x_ex.size(-1)
        with graph.inserting_before(node):
            x2d_size = _lift_int_or_symint_list(graph, [-1, k])
            x2d = graph.call_function(aten.reshape.default, (x, x2d_size))
            mm_node = graph.call_function(aten.mm.default, (x2d, w2d))
            if _is_tensor_meta(bmm_ex):
                out_size = _lift_int_or_symint_list(graph, list(bmm_ex.shape))
            else:
                out_size = _lift_int_or_symint_list(graph, [x_ex.size(0), x_ex.size(1), w_ex.size(-1)])
            out = graph.call_function(aten.reshape.default, (mm_node, out_size))
        try:
            _set_contiguous_meta(x2d, x_ex.reshape(-1, k))
        except (RuntimeError, ValueError, TypeError, IndexError):
            _copy_meta(x2d, x)
        if _is_tensor_meta(bmm_ex):
            try:
                _set_contiguous_meta(mm_node, bmm_ex.reshape(-1, bmm_ex.size(-1)))
            except (RuntimeError, ValueError, TypeError, IndexError):
                _copy_meta(mm_node, node)
            _copy_meta(out, node)
        else:
            _copy_meta(out, node)
        node.replace_all_uses_with(out)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()


def apply_torch_layout_contract_(gm: GraphModule) -> None:
    """Normalize FX layouts; leave consumer ABI adaptation to the C++ planner."""
    fold_consecutive_permutes_(gm)
    # Keep relative view operations until lowering knows the producer's actual
    # storage. FakeTensor can preserve a transposed layout for Add while the
    # selected OPP implementation writes row-major output. Baking the fake
    # absolute strides into as_strided would reinterpret that dense allocation.
    # Explicit as_strided and compiled-kernel argument layouts remain explicit.
    rewrite_broadcast_bmm_to_mm_(gm)
    # Consumer ABI adaptation belongs to C++ after the actual lowering Match.
    # Keep FX normalization/fusion here; do not write dense on consumer edges.
