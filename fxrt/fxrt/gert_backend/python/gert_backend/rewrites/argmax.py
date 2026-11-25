"""Graph rewrite: torch.argmax without a dim flattens before the reduce.

ArgMaxV2 takes its dimension as a rank-0 tensor input, so the Op.argmax row needs an
explicit axis; a dim=None aten call exports none and the lowering rejects the arity.
PyTorch defines dim=None as "argmax over the flattened input", so the rewrite emits
``argmax(reshape(x, (numel,)), -1)`` -- the numel is a product of the meta dims, which
keeps SymInt factors symbolic instead of pinning the example batch. A 0-d input
reshapes to (1,) and still reduces to the same scalar index.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.fx import GraphModule, Node

from fxrt.gert_backend.rewrites._common import example_of, shape_of
from fxrt.gert_backend.rewrites._framework import register_converter_graph_rewrite


def _stamp_meta(fx_node: Node, example: Any) -> None:
    """Stamp example_value/val meta on a freshly built node (same as as_strided's)."""
    fx_node.meta["example_value"] = example
    fx_node.meta["val"] = example


def _argmax_nodim_dim(node) -> Optional[Any]:
    """Return the dim argument of an argmax FX node (None when the call omits it)."""
    if "dim" in node.kwargs:
        return node.kwargs["dim"]
    return node.args[1] if len(node.args) > 1 else None


def _argmax_call_targets() -> set:
    """ATen and torch.argmax callables the nodim rewrite matches."""
    targets = {torch.argmax}
    for name in ("argmax", "argmax.default"):
        op = torch.ops.aten
        for part in name.split("."):
            op = getattr(op, part, None)
            if op is None:
                break
        if op is not None:
            targets.add(op)
    return targets


def _rewrite_argmax_nodim_(gm: GraphModule) -> None:
    """Rewrite dim=None argmax as flatten + argmax(-1) (Path C needs an explicit axis)."""
    targets = _argmax_call_targets()
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in targets:
            continue
        if _argmax_nodim_dim(node) is not None:
            continue
        if node.kwargs.get("keepdim", False):
            # Eager rejects keepdim without dim; leave the graph to fail the same way.
            continue
        shape = shape_of(node.args[0])
        if shape is None:
            continue
        numel: Any = 1
        for dim_size in shape:
            numel = numel * dim_size
        x_example = example_of(node.args[0])
        with graph.inserting_before(node):
            flat = graph.call_function(torch.reshape, args=(node.args[0], (numel,)))
            if isinstance(x_example, torch.Tensor):
                _stamp_meta(flat, x_example.reshape(numel))
            kwargs = {key: value for key, value in node.kwargs.items() if key != "dim"}
            repl = graph.call_function(node.target, args=(flat, -1), kwargs=kwargs)
            out_example = example_of(node)
            if out_example is not None:
                _stamp_meta(repl, out_example)
        node.replace_all_uses_with(repl)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()


# dim=None argmax becomes flatten + argmax(-1) before IR build; dim forms lower as-is.
register_converter_graph_rewrite(_rewrite_argmax_nodim_)
