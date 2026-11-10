"""Compose two dimension-zero indices across a flattening reshape.

Match index -> view/reshape -> index for a contiguous input of rank >= 2
and one-dimensional integer indices. The reshape merges every dimension
except the final feature dimension; the lowering composes flattened indices.
Symbolic sizes use range guards for safety and profitability without fixing
dimensions to the values observed during tracing.
The post-AOT rewrite supports inference and training. AOT's existing backward
graph is retained; the internal op is not inserted before differentiation.
"""

import operator
from math import prod

import torch
from torch._dynamo.utils import counters
from torch._inductor.pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true

from ... import config
from ..utils.fx_pass_level import PassType
from .register_custom_pass import register_custom_pass


torch.library.define(
    "npu_ext::index_reshape_index",
    "(Tensor input_tensor, Tensor first_indices, Tensor second_indices) -> Tensor",
)


def index_reshape_index_reference(input_tensor, first_indices, second_indices):
    first_index_result = torch.ops.aten.index.Tensor(input_tensor, [first_indices])
    flattened = torch.ops.aten.view.default(
        first_index_result,
        [
            first_indices.numel() * prod(input_tensor.shape[1:-1]),
            input_tensor.shape[-1],
        ],
    )
    return torch.ops.aten.index.Tensor(flattened, [second_indices])


torch.library.impl("npu_ext::index_reshape_index", "CompositeExplicitAutograd")(
    index_reshape_index_reference
)


TARGET = torch.ops.npu_ext.index_reshape_index.default
_patterns = PatternMatcherPass(pass_name="index_reshape_index")


def _is_index_reshape_index_profitable(input_tensor, first_indices, second_indices):
    """Conservative byte-volume heuristic with guards for symbolic sizes.

    Estimate the first index result, source, output, and index-buffer traffic.
    The 8 MiB output threshold and byte-volume ratios leave headroom for
    index-composition work. Backend division and launch latency are not modeled.
    """
    source_count = input_tensor.shape[0]
    block_size = prod(input_tensor.shape[1:-1])
    feature_size = input_tensor.shape[-1]
    first_count, second_count = first_indices.numel(), second_indices.numel()
    feature_bytes = feature_size * input_tensor.element_size()
    input_bytes = source_count * block_size * feature_bytes
    first_index_bytes = first_count * block_size * feature_bytes
    output_bytes = second_count * feature_bytes

    # Match the lowering: validate first-stage indices in INT64, then
    # use INT32 for composed indices only when the flattened range fits.
    flattened_rows = source_count * block_size
    # Symbolic ranges keep INT64 in lowering, even when the tracing hint fits.
    combined_index_bytes = (
        4 if type(flattened_rows) is int and flattened_rows <= 2**31 else 8
    )
    index_traffic_bytes = (
        first_count * (first_indices.element_size() + 8)
        + second_count * (second_indices.element_size() + 8 + 2 * combined_index_bytes)
    )
    # The first index result size is a cost estimate, not guaranteed traffic
    # savings. The second index reads and writes output in both implementations.
    return (
        guard_or_false(output_bytes >= 8 * 1024 * 1024)
        and guard_or_false(first_index_bytes >= 4 * input_bytes)
        and guard_or_false(first_index_bytes >= 4 * index_traffic_bytes)
    )


def _eligible(match):
    if not config.is_ascend950:
        return False
    tensors = [
        match.kwargs[name].meta.get("val")
        for name in ("input_tensor", "first_indices", "second_indices")
    ]
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    input_tensor, first_indices, second_indices = tensors
    if input_tensor.device.type != "npu" or any(
        t.device != input_tensor.device for t in tensors
    ):
        return False
    if input_tensor.ndim < 2 or first_indices.ndim != 1 or second_indices.ndim != 1:
        return False
    # Guard ranges, rather than specializing dimensions to their tracing hints.
    # Unbacked dimensions that cannot be proved or guarded retain the ATen path.
    if not all(guard_or_false(d > 0) for t in tensors for d in t.shape):
        return False
    block_size = prod(input_tensor.shape[1:-1])
    if not all(
        guard_or_false(count * block_size < 2**63)
        for count in (first_indices.numel(), input_tensor.shape[0])
    ):
        return False
    if not input_tensor.is_contiguous():
        return False
    if input_tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if any(
        t.dtype not in (torch.int32, torch.int64)
        for t in (first_indices, second_indices)
    ):
        return False
    view = match.output_node().args[0]
    # AOT may save intermediate sizes for backward. Permit those scalar users,
    # but retain the original graph if an intermediate tensor itself is shared.
    matched_nodes = set(match.nodes)
    for node in (view.args[0], view):
        for user in node.users:
            if user in matched_nodes:
                continue
            node_meta = node.meta.get("val")
            if not (
                isinstance(node_meta, torch.Tensor)
                and user.op == "call_function"
                and user.target == torch.ops.aten.sym_size.int
                and len(user.args) == 2
                and type(user.args[1]) is int
                and -node_meta.ndim <= user.args[1] < node_meta.ndim
            ):
                return False
    meta = view.meta.get("val")
    expected = (first_indices.numel() * block_size, input_tensor.shape[-1])
    # Only fuse a flatten whose shape is symbolically equivalent. A matching
    # concrete hint alone is insufficient evidence for a dynamic reshape.
    if not isinstance(meta, torch.Tensor) or meta.ndim != 2 or not all(
        statically_known_true(actual == wanted)
        for actual, wanted in zip(meta.shape, expected)
    ):
        return False
    if not guard_or_false(second_indices.numel() * input_tensor.shape[-1] < 2**63):
        return False
    return _is_index_reshape_index_profitable(
        input_tensor, first_indices, second_indices
    )


_pattern = CallFunction(
    torch.ops.aten.index.Tensor,
    CallFunction(
        (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ),
        CallFunction(
            torch.ops.aten.index.Tensor,
            KeywordArg("input_tensor"),
            [KeywordArg("first_indices")],
            _users=MULTIPLE,
        ),
        Ignored(),
        _users=MULTIPLE,
    ),
    [KeywordArg("second_indices")],
)


@register_graph_pattern(_pattern, extra_check=_eligible, pass_dict=_patterns)
def _replace(match, input_tensor, first_indices, second_indices):
    root = match.output_node()
    view = root.args[0]
    first_index = view.args[0]
    matched_nodes = set(match.nodes)

    def size(node, dim):
        result = match.graph.call_function(torch.ops.aten.sym_size.int, (node, dim))
        result.meta["val"] = node.meta["val"].shape[dim]
        return result

    for node in (first_index, view):
        for query in list(node.users):
            if query in matched_nodes:
                continue
            dim = query.args[1] % node.meta["val"].ndim
            with match.graph.inserting_before(query):
                if node is first_index:
                    replacement = (
                        size(first_indices, 0) if dim == 0 else size(input_tensor, dim)
                    )
                elif dim == 1:
                    replacement = size(input_tensor, input_tensor.meta["val"].ndim - 1)
                else:
                    replacement = size(first_indices, 0)
                    for axis in range(1, input_tensor.meta["val"].ndim - 1):
                        factor = size(input_tensor, axis)
                        product = match.graph.call_function(
                            operator.mul, (replacement, factor)
                        )
                        product.meta["val"] = (
                            replacement.meta["val"] * factor.meta["val"]
                        )
                        replacement = product
                replacement.meta = query.meta.copy()
            query.replace_all_uses_with(replacement)
            match.graph.erase_node(query)
    with match.graph.inserting_before(root):
        replacement = match.graph.call_function(
            TARGET, (input_tensor, first_indices, second_indices)
        )
        replacement.meta = root.meta.copy()
    root.replace_all_uses_with(replacement)
    match.erase_nodes()
    counters["inductor"]["npu_index_reshape_index"] += 1


@register_custom_pass(PassType.POST, ignore_inference_check=True)
def index_reshape_index_pass(graph):
    # This pass runs after AOT differentiation. Preserve graph outputs and reject
    # shared intermediate tensors through _eligible; scalar size users are rebuilt.
    if config.enable_full_lowering_fallback.strip() != "allfallback":
        _patterns.apply(graph)
