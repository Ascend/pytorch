"""Slice / getitem / select analysis shared by the arg, ops, and output hooks.

Parses aten slice / narrow / tensor-getitem / select FX nodes into normalized axes and
the ``(x, offsets, size)`` vectors the Slice family consumes. Static geometry is
kept as a storage view; unproven geometry retains the existing data path.
"""

from __future__ import annotations

import operator
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch.fx import Node

from fxrt.gert_backend.rewrites._common import (
    Axis,
    concrete_int,
    emit_slice,
    example_of,
    int_tuple_node,
    normalize_dim,
    offsets_from_axes,
    shape_of,
    strided_pieces,
)
from fxrt.ir import Op


def _slice_bounds(args: tuple, narrow: bool) -> Optional[Tuple[Any, int, int, Optional[int], int]]:
    """Return (self, dim, start, end, step). ``end`` is None for an open slice."""
    if len(args) < 4:
        return None
    dim = concrete_int(args[1])
    start = concrete_int(args[2])
    if dim is None or start is None:
        return None
    if narrow:
        length = concrete_int(args[3])
        if length is None:
            return None
        return args[0], dim, start, start + length, 1
    end_arg = args[3]
    end = None if end_arg is None else concrete_int(end_arg)
    if end_arg is not None and end is None:
        return None
    step = 1 if len(args) < 5 or args[4] is None else concrete_int(args[4])
    if step is None:
        return None
    return args[0], dim, start, end, step


def _slice_targets():
    """Aten and torch callables that this rewrite treats as slice or narrow."""
    names = []
    for name in ("slice", "narrow"):
        packet = getattr(torch.ops.aten, name, None)
        if packet is None:
            continue
        names.append(packet)
        default = getattr(packet, "default", None)
        tensor = getattr(packet, "Tensor", None)
        if default is not None:
            names.append(default)
        if tensor is not None:
            names.append(tensor)
    names.extend([getattr(torch, "slice", None), getattr(torch, "narrow", None), "slice", "narrow"])
    return {target for target in names if target is not None}


def _storage_root(node: Any) -> Any:
    """Walk recognized metadata views to the input whose data pointer anchors offsets."""
    view_methods = {"view", "reshape", "flatten", "squeeze", "unsqueeze", "t",
                    "transpose", "permute", "movedim", "as_strided", "select",
                    "narrow", "split", "chunk", "unbind"}
    view_targets = set()
    for name in ("view", "reshape", "_unsafe_view", "flatten", "squeeze", "unsqueeze",
                 "t", "transpose", "permute", "movedim", "as_strided", "select",
                 "narrow", "slice", "split", "split_with_sizes", "chunk", "unbind"):
        packet = getattr(torch.ops.aten, name, None)
        function = getattr(torch, name, None)
        if function is not None:
            view_targets.add(function)
        if packet is not None:
            view_targets.add(packet)
            for overload in getattr(packet, "overloads", lambda: [])():
                view_targets.add(getattr(packet, overload))
    while isinstance(node, Node):
        is_view = (node.op == "call_method" and node.target in view_methods) or (
            node.op == "call_function" and node.target in view_targets)
        if node.op == "call_function" and node.target is operator.getitem:
            # Tensor slicing and projection from a tuple of views both retain
            # the original storage coordinate system.
            is_view = getitem_axes(node) is not None or isinstance(
                example_of(node.args[0]), (tuple, list))
        if not is_view or not node.args or not isinstance(node.args[0], Node):
            return node
        source = example_of(node.args[0])
        result = example_of(node)
        if isinstance(source, torch.Tensor) and isinstance(result, torch.Tensor):
            # reshape/flatten may allocate. Do not subtract an external feed's
            # origin after a real storage boundary.
            if not torch._C._is_alias_of(source, result):
                return node
        node = node.args[0]
    return node


def _feed_storage_offset(node: Any) -> Optional[int]:
    """Return the converter feed's storage-coordinate origin (the storage base)."""
    root = _storage_root(node)
    if not isinstance(root, Node) or root.op != "placeholder":
        return 0
    return 0


def _narrow_targets():
    packet = getattr(torch.ops.aten, "narrow", None)
    targets = {torch.narrow, "narrow"}
    if packet is not None:
        targets.add(packet)
        default = getattr(packet, "default", None)
        if default is not None:
            targets.add(default)
    return targets


def slice_axes(node: Node) -> Optional[Tuple[Any, List[Any], List[Axis]]]:
    """Return ``(data, shape, axes)`` for aten slice / narrow."""
    if node.target not in _slice_targets():
        return None
    bounds = _slice_bounds(node.args, node.target in _narrow_targets())
    if bounds is None:
        return None
    data, dim, start, end, step = bounds
    shape = shape_of(data)
    if shape is None:
        return None
    norm_dim = normalize_dim(dim, len(shape))
    if norm_dim is None:
        return None
    return data, shape, [(norm_dim, start, end, step)]


def _axis_from_slice(dim: int, index: slice) -> Optional[Axis]:
    """Turn one Python slice into ``(dim, start, end, step)``, or None if symbolic."""
    start = 0 if index.start is None else concrete_int(index.start)
    if index.start is not None and start is None:
        return None
    end = None if index.stop is None else concrete_int(index.stop)
    if index.stop is not None and end is None:
        return None
    step = 1 if index.step is None else concrete_int(index.step)
    if step is None:
        return None
    return dim, start, end, step


def getitem_axes(node: Node) -> Optional[Tuple[Any, List[Any], List[Axis]]]:
    """Return ``(data, shape, axes)`` for a tensor getitem of slices."""
    if node.target is not operator.getitem or len(node.args) < 2:
        return None
    data = node.args[0]
    if not isinstance(data, Node) or not isinstance(example_of(data), torch.Tensor):
        return None
    shape = shape_of(data)
    if shape is None:
        return None
    index = node.args[1]
    if isinstance(index, slice):
        axis = _axis_from_slice(0, index)
        return None if axis is None else (data, shape, [axis])
    if not isinstance(index, tuple) or any(item is Ellipsis or item is None for item in index):
        return None
    axes: List[Axis] = []
    for dim, item in enumerate(index):
        if not isinstance(item, slice):
            return None
        axis = _axis_from_slice(dim, item)
        if axis is None:
            return None
        axes.append(axis)
    return data, shape, axes


def slice_input_vectors(node: Node) -> Optional[Tuple[Any, tuple, tuple]]:
    """Return ``(data, offsets, size)`` when every sliced axis has step 1."""
    parsed = slice_axes(node)
    if parsed is None:
        return None
    data, shape, axes = parsed
    if any(axis[3] != 1 for axis in axes):
        return None
    vectors = offsets_from_axes(shape, axes)
    if vectors is None:
        return None
    return data, vectors[0], vectors[1]


def getitem_input_vectors(node: Node) -> Optional[Tuple[Any, tuple, tuple]]:
    """Return ``(data, offsets, size)`` for a step-1 tensor getitem."""
    parsed = getitem_axes(node)
    if parsed is None:
        return None
    data, shape, axes = parsed
    if any(axis[3] != 1 for axis in axes):
        return None
    vectors = offsets_from_axes(shape, axes)
    if vectors is None:
        return None
    return data, vectors[0], vectors[1]


def emit_strided_view(node: Node, data_node, shape: Sequence[Any], axes: Sequence[Axis], executor, sym_mgr):
    """Represent a statically described slice as a storage view.

    Compose geometry from the source layout and slice bounds. Symbolic geometry
    keeps the existing graph-based fallback.
    """
    source = example_of(node.args[0])
    if not isinstance(source, torch.Tensor):
        return None
    sizes = [concrete_int(dim) for dim in shape]
    strides = [concrete_int(stride) for stride in source.stride()]
    offset = concrete_int(source.storage_offset())
    feed_offset = _feed_storage_offset(node.args[0])
    if offset is None or feed_offset is None or any(stride is None for stride in strides):
        return None
    offset -= feed_offset
    for dim, start, end, step in axes:
        if step <= 0:
            return None
        if sizes[dim] is None:
            # A dynamic extent with a fixed, guarded result size does not
            # require a symbolic offset/stride. Check real bounds per Execute.
            output_size = concrete_int(example_of(node).shape[dim])
            if start < 0 or output_size is None:
                return None
            sizes[dim] = output_size
        else:
            start, stop, step = slice(start, end, step).indices(sizes[dim])
            sizes[dim] = len(range(start, stop, step))
        offset += start * strides[dim]
        strides[dim] *= step
    if any(dim is None for dim in sizes):
        return None
    size_node = int_tuple_node(executor, sym_mgr, sizes)
    stride_node = int_tuple_node(executor, sym_mgr, strides)
    offset_node = executor.add_value_node(sym_mgr.from_torch_with_sym(offset))
    output = sym_mgr.from_torch_with_sym(example_of(node))
    result = executor.add_op_node(Op.as_strided_view, [data_node, size_node, stride_node, offset_node], output)
    # Relative slicing must follow the selected producer's physical layout,
    # which can differ from FakeTensor's preferred output strides.
    normalized = []
    for dim, start, end, step in axes:
        extent = concrete_int(shape[dim])
        if extent is None:
            return result
        first, stop, step = slice(start, end, step).indices(extent)
        normalized.extend((dim, first, len(range(first, stop, step)), step))
    result.set_attr("fxrt_slice_geometry", sym_mgr.from_torch_with_sym(tuple(normalized)))
    return result


def emit_strided_cat(node: Node, data_node, shape: Sequence[Any], axes: Sequence[Axis], executor, sym_mgr):
    """Concatenate length-1 slices for the single strided axis."""
    direct = emit_strided_view(node, data_node, shape, axes, executor, sym_mgr)
    if direct is not None:
        return direct
    plan = strided_pieces(shape, axes)
    if plan is None:
        return None
    dim, pieces = plan
    data_example = example_of(node.args[0])
    built = []
    for offsets, size in pieces:
        length = size[dim]
        if not isinstance(length, int):
            return None
        piece_example = data_example.narrow(dim, offsets[dim], length)
        built.append(emit_slice(executor, sym_mgr, data_node, offsets, size, piece_example))
    if len(built) == 1:
        return built[0]
    output = sym_mgr.from_torch_with_sym(example_of(node))
    dim_node = executor.add_value_node(sym_mgr.from_torch_with_sym(dim))
    return executor.add_op_node(Op.cat, built + [dim_node], output)


def select_axis_index(node: Node) -> Optional[Tuple[int, int]]:
    """Normalize positional and keyword select arguments without specializing sizes."""
    if not node.args:
        return None
    shape = shape_of(node.args[0])
    dim = concrete_int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None))
    index = concrete_int(node.kwargs.get("index", node.args[2] if len(node.args) > 2 else None))
    if shape is None or dim is None or index is None:
        return None
    norm_dim = normalize_dim(dim, len(shape))
    if norm_dim is None:
        return None
    return norm_dim, index


def emit_select(node: Node, data_node, executor, sym_mgr):
    """Emit select as a storage view with the selected axis removed."""
    parsed = select_axis_index(node)
    if parsed is None:
        return None
    norm_dim, index = parsed
    shape = shape_of(node.args[0])
    extent = shape[norm_dim]
    if isinstance(extent, int) and not isinstance(extent, bool) and index < 0:
        index += extent
    source = example_of(node.args[0])
    if not isinstance(source, torch.Tensor):
        return None
    sizes = [concrete_int(dim) for dim in shape]
    strides = [concrete_int(stride) for stride in source.stride()]
    offset = concrete_int(source.storage_offset())
    feed_offset = _feed_storage_offset(node.args[0])
    if offset is None or feed_offset is None or any(stride is None for stride in strides):
        return None
    offset -= feed_offset
    if index < 0 or (sizes[norm_dim] is not None and index >= sizes[norm_dim]):
        return None
    offset += index * strides[norm_dim]
    del sizes[norm_dim]
    del strides[norm_dim]
    if any(dim is None for dim in sizes):
        return None
    result_example = example_of(node)
    size_node = int_tuple_node(executor, sym_mgr, sizes)
    stride_node = int_tuple_node(executor, sym_mgr, strides)
    offset_node = executor.add_value_node(sym_mgr.from_torch_with_sym(offset))
    output = sym_mgr.from_torch_with_sym(result_example)
    result = executor.add_op_node(Op.as_strided_view, [data_node, size_node, stride_node, offset_node], output)
    result.set_attr("fxrt_select_geometry", sym_mgr.from_torch_with_sym((norm_dim, index)))
    return result
