"""Rebuild the callable's return value from GE's flat tensor list.

The compiled session is a GE C graph: execute() returns only the materialized
tensor buffers, and everything else the FX output node carries -- nesting, SymInt
or Python scalar items, and torch-semantics dtypes -- is lost at the compile
boundary. This module interprets the FX output tree to rebuild what Dynamo
returned:

- tensor items consume the flat list in order; a dtype mismatch against the FX
  example (GE TopKV2 writes int32 where torch returns int64) is cast back;
- non-tensor items (SymInt such as ``4 * (s72 // 4)``) are replayed from the call
  args, since their scalar subgraph depends only on placeholders;
- arbitrary tuple/list nesting is preserved.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch.fx import GraphModule, Node

from fxrt import fx_backend as fxb


# fx_backend's _get_node_meta_value is the same meta the IR builder uses to decide
# which returns are ModelV2 tensors.
# pylint: disable=protected-access


def fx_output_tree(gm: GraphModule) -> Any:
    """The FX output node's argument tree: what Dynamo expects the callable to return."""
    for node in gm.graph.nodes:
        if node.op == "output":
            return node.args[0] if node.args else ()
    raise RuntimeError("fx_converter_backend: FX graph has no output node")


def _example_is_tensor(item: Any) -> bool:
    """Whether the FX output item is, or wraps, a tensor example value."""
    if torch.is_tensor(item):
        return True
    if isinstance(item, Node):
        return torch.is_tensor(fxb._get_node_meta_value(item))
    return False


def _eval_fx_arg(
    arg: Any,
    call_args: Sequence[Any],
    placeholders: Sequence[Node],
    cache: Dict[Node, Any],
) -> Any:
    """Value of a non-tensor FX output arg; Node inputs are replayed from call args."""
    if isinstance(arg, Node):
        return _eval_fx_scalar_node(arg, call_args, placeholders, cache)
    if isinstance(arg, (tuple, list)):
        return type(arg)(_eval_fx_arg(item, call_args, placeholders, cache) for item in arg)
    if isinstance(arg, slice):
        start = None if arg.start is None else _eval_fx_arg(arg.start, call_args, placeholders, cache)
        stop = None if arg.stop is None else _eval_fx_arg(arg.stop, call_args, placeholders, cache)
        step = None if arg.step is None else _eval_fx_arg(arg.step, call_args, placeholders, cache)
        return slice(start, stop, step)
    return arg


def _eval_fx_scalar_node(
    node: Node,
    call_args: Sequence[Any],
    placeholders: Sequence[Node],
    cache: Dict[Node, Any],
) -> Any:
    """Evaluate a non-tensor FX output (SymInt / Python scalar) from call args.

    Dynamo may return values such as ``4 * (s72 // 4)`` alongside hidden states.
    Convert/GE only materialize tensors, so replay the scalar subgraph here.
    """
    if node in cache:
        return cache[node]
    if _example_is_tensor(node) and node.op != "placeholder":
        raise RuntimeError(
            "fx_converter_backend: scalar output depends on an intermediate tensor "
            f"({node.name})"
        )
    if node.op == "placeholder":
        try:
            idx = placeholders.index(node)
        except ValueError as exc:
            raise RuntimeError(
                f"fx_converter_backend: scalar output placeholder {node.name} is missing"
            ) from exc
        if idx >= len(call_args):
            raise RuntimeError(
                f"fx_converter_backend: placeholder {node.name} index {idx} exceeds args"
            )
        cache[node] = call_args[idx]
        return cache[node]
    eval_args = [_eval_fx_arg(arg, call_args, placeholders, cache) for arg in node.args]
    eval_kwargs = {
        key: _eval_fx_arg(val, call_args, placeholders, cache) for key, val in node.kwargs.items()
    }
    if node.op == "call_function":
        cache[node] = node.target(*eval_args, **eval_kwargs)
        return cache[node]
    if node.op == "call_method":
        cache[node] = getattr(eval_args[0], node.target)(*eval_args[1:], **eval_kwargs)
        return cache[node]
    raise RuntimeError(
        f"fx_converter_backend: cannot eval scalar FX node op={node.op} target={node.target}"
    )


class _OutputPacker:
    """Per-call output state without a self-referencing recursive closure.

    A recursive local function retains its own closure until cyclic GC runs.
    That closure also retains the flat iterator and input arguments, so even
    discarded outputs can hold device allocations across many Execute calls.
    """

    def __init__(self, flat, call_args, placeholders):
        self.flat_iter = iter(flat)
        self.call_args = call_args
        self.placeholders = placeholders
        self.cache = {}
        self.consumed = 0

    def pack(self, item: Any) -> Any:
        """Consume flat tensors in order, rebuilding the FX output tree."""
        if _example_is_tensor(item):
            try:
                tensor = next(self.flat_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "fx_converter_backend: fewer tensor outputs than FX tree expects"
                ) from exc
            self.consumed += 1
            example = fxb._get_node_meta_value(item) if isinstance(item, Node) else item
            expected_dtype = example.dtype if torch.is_tensor(example) else None
            if expected_dtype is not None and tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            return tensor
        if isinstance(item, (tuple, list)):
            return type(item)(self.pack(child) for child in item)
        if isinstance(item, Node):
            return _eval_fx_scalar_node(item, self.call_args, self.placeholders, self.cache)
        return item


def pack_like_fx(
    flat: Sequence[torch.Tensor],
    tree: Any,
    call_args: Sequence[Any],
    placeholders: Sequence[Node],
) -> Any:
    """Rebuild return value with the same nesting as the FX output node."""
    packer = _OutputPacker(flat, call_args, placeholders)
    packed = packer.pack(tree)
    if packer.consumed != len(flat):
        raise RuntimeError(
            "fx_converter_backend: more tensor outputs than FX tree expects "
            f"(got {len(flat)}, need {packer.consumed})"
        )
    return packed
