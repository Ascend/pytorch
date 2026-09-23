"""Implementations for torch setitem operation processing."""

from typing import Tuple
import torch
from torch.fx.node import Node
from fxrt.ir import Op
from fxrt.utils import tuple_indices_to_slice_arg


def tensor_setitem_slice_tensor(x: Node, indices: slice, value: Node):
    """Handle tensor_setitem_slice_tensor node."""
    # operation: x[indices] = value
    if hasattr(x, "meta") and x.meta.get("example_value") is not None:
        shape = x.meta["example_value"].shape
    else:
        raise ValueError("For 'tensor_setitem_slice_tensor', the first input must be a tensor.")

    begin = [indices.start if indices.start is not None else 0]
    end = [indices.stop if indices.stop is not None else shape[0]]
    strides = [indices.step if indices.step is not None else 1]
    axes = [0]

    return Op.strided_slice_assign, [x, value, begin, end, strides, axes]

def tensor_setitem_tuple_tensor(x: Node, indices: Tuple, value: Node):
    """Handle tensor_setitem_tuple_tensor node."""
    # operation: scatter_nd_update(x, index, value)
    if hasattr(x, "meta") and x.meta.get("example_value") is not None:
        shape = x.meta["example_value"].shape
    else:
        raise ValueError("For 'tensor_setitem_tuple_tensor', the first input must be a tensor.")

    begin, end, strides, axes = tuple_indices_to_slice_arg(indices, shape)
    return Op.strided_slice_assign, [x, value, begin, end, strides, axes]

# pylint: disable=unused-argument
def setitem_process(node, input_nodes):
    """Handle setitem node."""
    is_tensor_node = isinstance(input_nodes[0], torch.fx.node.Node)
    example_val = input_nodes[0].meta.get("example_value") if is_tensor_node else None
    # pylint: disable=protected-access
    if is_tensor_node and (input_nodes[0].type == torch.Tensor or
                           isinstance(example_val, torch._subclasses.FakeTensor)):
        if isinstance(input_nodes[1], slice):
            return tensor_setitem_slice_tensor(input_nodes[0], input_nodes[1], input_nodes[2])
        if isinstance(input_nodes[1], (list, tuple)):
            return tensor_setitem_tuple_tensor(input_nodes[0], input_nodes[1], input_nodes[2])
        return Op.python_call, [input_nodes[0], input_nodes[1], input_nodes[2]]
    raise ValueError("For 'setitem', the first input must be a tensor.")
