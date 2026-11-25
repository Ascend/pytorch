"""Restore torch's int64 TopK indices before graph consumers read them.

The TopKV2 catalog selects bins that write int32 indices. Widening only model
outputs leaves internal consumers, including compiled AutoFuse kernels, reading
an int32 allocation with the int64 ABI declared by the original FX graph.
"""

from __future__ import annotations

import operator

import torch
from torch.fx import GraphModule, Node

from fxrt.gert_backend.rewrites._common import example_of
from fxrt.gert_backend.rewrites._framework import register_converter_graph_rewrite


def _rewrite_topk_indices_(gm: GraphModule) -> None:
    targets = {torch.topk, torch.ops.aten.topk, torch.ops.aten.topk.default}
    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not operator.getitem or len(node.args) != 2:
            continue
        source, index = node.args
        if (index not in (1, -1) or not isinstance(source, Node) or source.op != "call_function"
                or source.target not in targets):
            continue
        expected = example_of(node)
        if not isinstance(expected, torch.Tensor) or expected.dtype != torch.int64:
            continue
        users = tuple(node.users)
        with gm.graph.inserting_after(node):
            widened = gm.graph.call_function(
                torch.ops.prims.convert_element_type.default,
                args=(node, torch.int64),
            )
        widened.meta = dict(node.meta)
        widened.meta.update(val=expected, example_value=expected)
        # Describe the actual TopKV2 allocation rather than torch's declared
        # int64 result. A second application then leaves this cast unchanged.
        native = expected.to(dtype=torch.int32)
        node.meta.update(val=native, example_value=native)
        for user in users:
            user.replace_input_with(node, widened)
        changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()


register_converter_graph_rewrite(_rewrite_topk_indices_)
