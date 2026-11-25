"""Lower safe softmax to native operations, preserving fully masked rows."""

from __future__ import annotations

import torch
from torch.fx import GraphModule
from torch.fx.node import map_arg

from fxrt.gert_backend.rewrites._common import example_of
from fxrt.gert_backend.rewrites._framework import register_converter_graph_rewrite


def _decompose_safe_softmax_(gm: GraphModule) -> None:
    """Keep dtype conversion and zero results for rows containing only -inf.

    NaN and +inf rows retain ordinary softmax semantics. A row maximum of -inf
    identifies fully masked rows without an integer cast and sum reduction.
    """
    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten._safe_softmax.default:
            continue
        sample, dim = node.args[:2]
        dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
        value = example_of(sample)
        if not isinstance(value, torch.Tensor):
            continue

        def emit(target, args, kwargs=None):
            kwargs = kwargs or {}
            result = graph.call_function(target, args, kwargs)
            example = target(*map_arg(args, example_of), **map_arg(kwargs, example_of))
            result.meta.update(val=example, example_value=example)
            return result

        with graph.inserting_before(node):
            softmax_input = sample
            if dtype is not None and dtype != value.dtype:
                softmax_input = emit(torch.ops.prims.convert_element_type.default, (sample, dtype))
            probabilities = emit(torch.ops.aten.softmax.int, (softmax_input, dim, None))
            row_max = emit(torch.ops.aten.amax.default, (sample, [dim], True))
            all_masked = emit(torch.ops.aten.eq.Scalar, (row_max, float("-inf")))
            output_dtype = example_of(probabilities).dtype
            if output_dtype == torch.bfloat16:
                # The installed Fills .o has no BF16 binary. A boolean cast is
                # finite even for NaN/inf input, so multiplying it by zero is safe.
                zero_seed = emit(torch.ops.prims.convert_element_type.default, (all_masked, output_dtype))
                zeros = emit(torch.ops.aten.mul.Scalar, (zero_seed, 0.0))
            else:
                zero_seed = row_max
                if example_of(row_max).dtype != output_dtype:
                    zero_seed = emit(torch.ops.prims.convert_element_type.default, (row_max, output_dtype))
                zeros = emit(torch.ops.aten.full_like.default, (zero_seed, 0.0))
            result = emit(torch.ops.aten.where.self, (all_masked, zeros, probabilities))
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()


register_converter_graph_rewrite(_decompose_safe_softmax_)
