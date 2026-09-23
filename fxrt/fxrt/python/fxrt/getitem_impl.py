"""Implementations for torch getitem operation processing."""

from typing import Tuple
import torch
from torch.fx.node import Argument, Node
from torch._subclasses.fake_tensor import FakeTensor
from fxrt.ir import Op
from fxrt.utils import tuple_indices_to_slice_arg, _dim_size_or_open

def _tensor_getitem_by_tuple(x: Node, indices: Tuple[Argument, ...]):
    """Convert getitem indices to getitem_slice parameters."""
    # x[1, ..., 1:10:2] -> getitem_slice(x, (1, ellipsis, slice(1, 10, 2)))
    shape = x.meta["example_value"].shape
    begin, end, steps, axes = tuple_indices_to_slice_arg(indices, shape)
    # Rewrite dynamic open ends to INT64_MAX so ABI0 int_list stays concrete Ints.
    end = [_dim_size_or_open(e) if not isinstance(e, int) else e for e in end]
    begin = [int(b) if not isinstance(b, int) else b for b in begin]
    return Op.getitem_slice, [x, begin, end, axes, steps]

def _tensor_getitem_by_slice(x: Node, indices: slice):
    """Convert getitem indices to getitem_slice parameters."""
    # x[1:10:2] -> getitem_slice(x, slice(1, 10, 2))
    shape = x.meta["example_value"].shape
    start = indices.start if indices.start is not None else 0
    if indices.stop is None:
        end = _dim_size_or_open(shape[0])
    else:
        end = indices.stop if isinstance(indices.stop, int) else _dim_size_or_open(indices.stop)
    step = indices.step if indices.step is not None else 1
    return Op.getitem_slice, [x, [start], [end], [0], [step]]

def _tensor_getitem_by_number(x: Node, indices: int):
    """Convert getitem indices to getitem_slice parameters."""
    # x[1] -> getitem_slice(x, [1], [2], [0], [1]) plus full ranges on trailing dims.
    shape = x.meta["example_value"].shape
    starts = [indices]
    ends = [indices + 1]
    axes = [0]
    steps = [1]
    for idx, dim_size in enumerate(shape[1:]):
        starts.append(0)
        ends.append(_dim_size_or_open(dim_size))
        axes.append(idx + 1)
        steps.append(1)
    return Op.getitem_slice, [x, starts, ends, axes, steps]

def _tensor_getitem_by_tensor(x: Node, indices: torch.Tensor):
    """Convert getitem indices to gather_v2 parameters."""
    # x[tensor] -> gather_v2(x, 0, tensor)
    return Op.gather_v2, [x, 0, indices]

def tuple_getitem(x, indices):
    """Handle tuple_getitem node."""
    # operation: Op.tuple_getitem(x, indices)
    return Op.tuple_getitem, [x, indices]


def _is_tensor_node(node):
    """Check if an FX node represents a tensor (by .type or example_value being FakeTensor)."""
    if not isinstance(node, torch.fx.node.Node):
        return False
    if node.type == torch.Tensor:
        return True
    example_val = node.meta.get("example_value")
    return isinstance(example_val, FakeTensor)


# pylint: disable=unused-argument
def getitem_process(node, input_nodes):
    """Handle getitem node."""
    if isinstance(input_nodes[0], (list, tuple)):
        return tuple_getitem(input_nodes[0], input_nodes[1])

    if isinstance(input_nodes[0], torch.fx.node.Node) and \
       (isinstance(input_nodes[0].meta.get("example_value"), (list, tuple)) or
       isinstance(input_nodes[0].meta.get("val"), (list, tuple))):
        return tuple_getitem(input_nodes[0], input_nodes[1])

    # input is tensor
    if _is_tensor_node(input_nodes[0]):
        idx_type = type(input_nodes[1])
        if idx_type is int:
            return _tensor_getitem_by_number(input_nodes[0], input_nodes[1])
        if idx_type is slice:
            return _tensor_getitem_by_slice(input_nodes[0], input_nodes[1])
        if idx_type is tuple:
            return _tensor_getitem_by_tuple(input_nodes[0], input_nodes[1])
        if isinstance(input_nodes[1], torch.fx.node.Node):
            return _tensor_getitem_by_tensor(input_nodes[0], input_nodes[1])
        input_node = input_nodes[0]
        index_node = input_nodes[1]
        node_name = getattr(input_node, "name", None)
        raise ValueError(
            "getitem indices type unsupported."
            f"Expected index types: int, slice, tuple, or Node. "
            f"Actual index type: {idx_type}. "
            f"node={node_name}, input_type={type(input_node)}, index_value={index_node}"
        )
    input0 = input_nodes[0]
    if isinstance(input0, torch.fx.node.Node):
        node_type = input0.type
        example_val = input0.meta.get("example_value")
        example_val_type = type(example_val)
    else:
        node_type = None
        example_val_type = None
    raise ValueError(
        "getitem input[0] unsupported. "
        "Expected: list, tuple, or Node (with type==Tensor or FakeTensor example_value). "
        f"Actual input[0] type: {type(input0)}, value: {input0}, "
        f"node_type={node_type}, example_value_type={example_val_type}"
    )
