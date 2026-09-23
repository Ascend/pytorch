"""
FX graph pass for decomposing aten.full.default.

FXRT already supports creating zero tensors and in-place scalar fill. This pass
rewrites aten.full.default to that supported op sequence before backend lowering,
so the common call-node lowering path can handle it without an aten.full-specific
branch.
"""

import torch
from torch.fx.graph_module import GraphModule


def _is_aten_full_default(node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.full.default


def _build_zeros_kwargs(node) -> dict:
    """Build torch.zeros keyword arguments from an aten.full.default node."""
    kwargs = {}
    example_value = node.meta.get("example_value", None)

    dtype = node.kwargs.get("dtype", None)
    if dtype is None and isinstance(example_value, torch.Tensor):
        dtype = example_value.dtype
    if dtype is not None:
        kwargs["dtype"] = dtype

    device = node.kwargs.get("device", None)
    if device is None and isinstance(example_value, torch.Tensor):
        device = example_value.device
    if device is not None:
        kwargs["device"] = device

    for key in ("layout", "pin_memory"):
        value = node.kwargs.get(key, None)
        if value is not None:
            kwargs[key] = value

    return kwargs


def decompose_full_(gm: GraphModule) -> None:
    """
    Replace aten.full.default(size, fill_value, ...) with:
      zeros(size, ...).fill_(fill_value)
    """
    graph = gm.graph
    full_nodes = [node for node in graph.nodes if _is_aten_full_default(node)]
    if not full_nodes:
        return

    for full_node in full_nodes:
        if len(full_node.args) < 2:
            continue

        size = full_node.args[0]
        fill_value = full_node.args[1]

        with graph.inserting_before(full_node):
            zeros_node = graph.call_function(torch.zeros, (size,), _build_zeros_kwargs(full_node))
            zeros_node.meta = full_node.meta.copy()
            fill_node = graph.call_method("fill_", (zeros_node, fill_value), {})
            fill_node.meta = full_node.meta.copy()

        full_node.replace_all_uses_with(fill_node)
        graph.erase_node(full_node)

    graph.lint()
    gm.recompile()
