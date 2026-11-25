"""Helpers shared by the converter rewrite categories (indexing / pad / reduce).

Everything here is pure analysis on FX nodes and example values: resolving concrete
ints and shapes, normalizing slice axes into ``offsets`` / ``size`` vectors, and
emitting int-tuple values and Slice nodes into the IR executor.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch.fx import Node

from fxrt.ir import Op
from fxrt import fx_backend as fxb

_INT64_MAX = 2 ** 63 - 1

# One normalized axis: (dim, start, end, step); end is None for an open slice.
Axis = Tuple[int, int, Optional[int], int]


def example_of(value: Any) -> Any:
    if not isinstance(value, Node):
        return value
    example = fxb._get_node_meta_value(value)  # pylint: disable=protected-access
    if example is None and "val" in value.meta:
        example = value.meta["val"]
    return example


def shape_of(tensor_node: Any) -> Optional[List[Any]]:
    example = example_of(tensor_node)
    shape = getattr(example, "shape", None)
    if shape is None:
        return None
    return list(shape)


def concrete_int(value: Any) -> Optional[int]:
    """Return a Python int. Symbolic and non-numeric values decline the rewrite.

    A Dynamo-specialized SymInt (``Sym(0)`` -- its expr is an Integer, e.g. a closed-over
    dim lifted as a SymInt placeholder) is a constant, not a real symbol: read it without
    adding a value guard. Only a genuinely symbolic expr declines.
    """
    resolved = example_of(value)
    if isinstance(resolved, torch.SymInt):
        expr = resolved.node.expr
        return int(expr) if expr.is_Integer else None
    if isinstance(resolved, bool):
        return None
    if isinstance(resolved, int):
        return resolved
    return None


def dim_extent(dim_size: Any) -> Any:
    """Keep a SymInt extent. A missing extent declines the rewrite."""
    if isinstance(dim_size, bool):
        return None
    if isinstance(dim_size, (int, torch.SymInt)):
        return dim_size
    return None


def normalize_dim(dim: int, rank: int) -> Optional[int]:
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    return dim


def clamp_concrete_range(start: int, end: int, dim_size: int) -> Tuple[int, int]:
    """Match Slice axis normalization for a concrete dimension length."""
    if end == _INT64_MAX or end > dim_size:
        end = dim_size
    if start < 0:
        start += dim_size
    if end < 0:
        end += dim_size
    start = min(max(start, 0), dim_size)
    end = min(max(end, 0), dim_size)
    end = max(end, start)
    return start, end


def axis_offset_and_length(start: int, end: Optional[int], dim_size: Any) -> Optional[Tuple[Any, Any]]:
    """Return (offset, length) for one step-1 axis, or None when it cannot be formed."""
    if isinstance(dim_size, int) and not isinstance(dim_size, bool):
        resolved_end = dim_size if end is None else end
        offset, resolved_end = clamp_concrete_range(start, resolved_end, dim_size)
        return offset, resolved_end - offset
    if start == 0 and end is None:
        return 0, dim_size
    if end is None or end == _INT64_MAX or start < 0:
        return None
    return start, end - start


def offsets_from_axes(shape: Sequence[Any], axes: Sequence[Axis]) -> Optional[Tuple[tuple, tuple]]:
    """Build full-rank offsets and size. Every axis must already be step 1."""
    by_dim = {axis[0]: axis for axis in axes}
    offsets: List[Any] = []
    sizes: List[Any] = []
    for index, extent in enumerate(shape):
        kept = dim_extent(extent)
        if kept is None:
            return None
        axis = by_dim.get(index)
        if axis is None:
            offsets.append(0)
            sizes.append(kept)
            continue
        if axis[3] != 1:
            return None
        piece = axis_offset_and_length(axis[1], axis[2], kept)
        if piece is None:
            return None
        offsets.append(piece[0])
        sizes.append(piece[1])
    return tuple(offsets), tuple(sizes)


def enumerate_indices(start: int, end: Optional[int], step: int, dim_size: Any) -> Optional[List[int]]:
    """Expand one concrete ``step != 1`` axis into indexes. Empty means the rewrite declines."""
    if step <= 0:
        return None
    if isinstance(dim_size, int) and not isinstance(dim_size, bool):
        resolved_end = dim_size if end is None else end
        start, resolved_end = clamp_concrete_range(start, resolved_end, dim_size)
    elif end is None or start < 0 or end < 0:
        return None
    else:
        resolved_end = end
    if resolved_end < start:
        return None
    return list(range(start, resolved_end, step))


def strided_pieces(shape: Sequence[Any], axes: Sequence[Axis]) -> Optional[Tuple[int, List[Tuple[tuple, tuple]]]]:
    """One ``step != 1`` axis becomes length-1 slices. Other axes stay step 1."""
    stepped = [axis for axis in axes if axis[3] != 1]
    if len(stepped) != 1:
        return None
    dim, start, end, step = stepped[0]
    extent = dim_extent(shape[dim]) if dim < len(shape) else None
    if extent is None:
        return None
    indices = enumerate_indices(start, end, step, extent)
    if not indices:
        return None
    pieces: List[Tuple[tuple, tuple]] = []
    for index in indices:
        rewritten = [(dim, index, index + 1, 1) if axis[0] == dim else axis for axis in axes]
        vectors = offsets_from_axes(shape, rewritten)
        if vectors is None:
            return None
        pieces.append(vectors)
    return dim, pieces


def concretize_shape_dim(dim: Any) -> Any:
    """Keep static Python ints; hand SymInt dims to the backend as symbols.

    ``int(symint)`` installs equality guards (e.g. batch==120). fx_converter_backend
    runs inside inductor compile_graph, so that breaks RecSDK ``mark_dynamic``:
    ConstraintViolationError if relax is off, or a new ModelV2 per batch if on.

    A SymInt is therefore returned as-is: ``_map_arg`` runs it through
    ``SymbolicShapeManager.from_torch_with_sym``, which builds an ``ir.Value`` holding the
    symbolic expression, and the ABI0 bridge carries it (ValueExport::Kind::kSymbol ->
    mrt_ir_graph_builder rebuilds a SymbolicVar). The converter then evaluates that
    expression per Execute against the same seeds InferShape uses.

    This deliberately does NOT substitute -1 for a dynamic dim. A -1 in a size list means
    what PyTorch means by it -- "infer this one", as in ``reshape(x, [-1])`` -- and a
    sentinel that collides with it leaves the runtime unable to tell an inferred dim from
    a dynamic one, with nothing left to do but guess a batch from the feed shapes.
    """
    if isinstance(dim, bool):
        return dim
    if isinstance(dim, int):
        return dim
    if isinstance(dim, torch.SymInt):
        return dim
    if isinstance(dim, Node):
        example = example_of(dim)
        if isinstance(example, bool):
            return dim
        if isinstance(example, int):
            return example
        if isinstance(example, torch.SymInt):
            # The SymInt itself, not the node: a size list wants a value, and the node
            # would come back as a graph edge that the ABI0 int_list cannot hold.
            return example
    return dim


def concretize_int_sequence(arg: Any) -> Any:
    """Concretize a size/stride list, Size, or FX Node producing an int sequence."""
    if isinstance(arg, (list, tuple)):
        return type(arg)(concretize_shape_dim(d) for d in arg)
    if isinstance(arg, torch.Size):
        return tuple(concretize_shape_dim(d) for d in arg)
    if isinstance(arg, Node):
        example = example_of(arg)
        if isinstance(example, (list, tuple, torch.Size)):
            return tuple(concretize_shape_dim(d) for d in example)
    return concretize_shape_dim(arg)


def int_tuple_node(executor, sym_mgr, values: Sequence[Any]):
    elements = [executor.add_value_node(sym_mgr.from_torch_with_sym(value)) for value in values]
    return executor.make_tuple(elements)


def emit_slice(executor, sym_mgr, data_node, offsets: tuple, size: tuple, example):
    offset_node = int_tuple_node(executor, sym_mgr, offsets)
    size_node = int_tuple_node(executor, sym_mgr, size)
    output = sym_mgr.from_torch_with_sym(example)
    return executor.add_op_node(Op.slice, [data_node, offset_node, size_node], output)


def node_output(node: Node, sym_mgr):
    return sym_mgr.from_torch_with_sym(example_of(node))
