"""Normalize pure Python new-axis indexing to metadata-only unsqueeze nodes."""

import operator

import torch
from torch.fx import GraphModule, Node

from fxrt.gert_backend.rewrites._common import example_of
from fxrt.gert_backend.rewrites._framework import register_converter_graph_rewrite


def _rewrite_newaxis_(gm: GraphModule) -> None:
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not operator.getitem or len(node.args) != 2:
            continue
        source, index = node.args
        sample = example_of(source)
        if not isinstance(source, Node) or not isinstance(sample, torch.Tensor):
            continue
        indices = index if isinstance(index, tuple) else (index,)
        if not any(item is None for item in indices):
            continue
        if any(item is not None and not (isinstance(item, slice) and item == slice(None)) for item in indices):
            continue
        if sum(item is not None for item in indices) > sample.ndim:
            continue
        replacement = source
        value = sample
        with graph.inserting_before(node):
            for axis, item in enumerate(indices):
                if item is None:
                    replacement = graph.call_function(torch.ops.aten.unsqueeze.default, (replacement, axis))
                    value = value.unsqueeze(axis)
                    replacement.meta.update(val=value, example_value=value)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()


register_converter_graph_rewrite(_rewrite_newaxis_)
