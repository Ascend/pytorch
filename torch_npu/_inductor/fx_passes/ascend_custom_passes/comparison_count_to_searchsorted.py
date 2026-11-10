"""Replace broadcast comparison counts with sorted reference searches.

Sorting a private reference copy preserves counts even for unordered inputs.
Consumers of the original reference_values retain their original indexing semantics.
This post-AOT rewrite does not assume ordered reference inputs.
"""

import operator

import torch
from torch._dynamo.utils import counters
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true

from ... import config
from ..utils.fx_pass_level import PassType
from .register_custom_pass import register_custom_pass


aten = torch.ops.aten


def _arg(node, position, name, default=None):
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(name, default)


def _is_call(node, target):
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target == target
    )


def _shape(node):
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, FakeTensor):
        return None
    return tuple(value.shape)


def _same_shape(actual, expected):
    # Layout relations must be proved without specializing symbolic dimensions.
    return (
        actual is not None
        and len(actual) == len(expected)
        and all(statically_known_true(a == b) for a, b in zip(actual, expected))
    )


def _strip_full_slices(node):
    # Python values[:, None] may leave full slices around the broadcast view.
    while _is_call(node, aten.slice.Tensor):
        source = _arg(node, 0, "self")
        shape = _shape(source)
        dim = _arg(node, 1, "dim", 0)
        start = _arg(node, 2, "start", 0)
        end = _arg(node, 3, "end", 9223372036854775807)
        step = _arg(node, 4, "step", 1)
        if (
            shape is None
            or not shape
            or type(dim) is not int
            or not -len(shape) <= dim < len(shape)
            or type(start) is not int
            or start != 0
            or type(end) is not int
            or (
                end != 9223372036854775807
                and not statically_known_true(end >= shape[dim])
            )
            or type(step) is not int
            or step != 1
        ):
            break
        node = source
    return node


def _broadcast_vector(node, axis):
    """Recover a vector varying on axis from a singleton view/expand chain."""
    node = _strip_full_slices(node)
    while isinstance(node, torch.fx.Node):
        source = _arg(node, 0, "self")
        if _is_call(node, aten.expand.default):
            node = _strip_full_slices(source)
            continue
        if _is_call(node, aten.permute.default):
            dims = _arg(node, 1, "dims")
            if (
                _shape(source) is None
                or len(_shape(source)) != 2
                or not isinstance(dims, (tuple, list))
                or len(dims) != 2
                or any(type(dim) is not int for dim in dims)
                or any(dim not in (0, 1, -1, -2) for dim in dims)
                or {dim % 2 for dim in dims} != {0, 1}
            ):
                return None
            axis = dims[axis] % 2
            node = _strip_full_slices(source)
            continue
        if _is_call(node, aten.transpose.int):
            dims = (_arg(node, 1, "dim0"), _arg(node, 2, "dim1"))
            if (
                _shape(source) is None
                or len(_shape(source)) != 2
                or any(type(dim) is not int for dim in dims)
                or any(dim not in (0, 1, -1, -2) for dim in dims)
            ):
                return None
            if dims[0] % 2 != dims[1] % 2:
                axis = 1 - axis
            node = _strip_full_slices(source)
            continue
        break
    shape = _shape(node)
    if (
        shape is None
        or len(shape) != 2
        or not statically_known_true(shape[1 - axis] == 1)
    ):
        return None
    if not any(
        _is_call(node, target)
        for target in (aten.unsqueeze.default, aten.view.default, aten.reshape.default)
    ):
        return None
    source = _arg(node, 0, "self")
    if _same_shape(_shape(source), shape):
        # Repeated singleton views preserve the same vector axis.
        return _broadcast_vector(source, axis)
    if not _same_shape(_shape(source), (shape[axis],)):
        return None
    return source


def _profitable(value_count, reference_count, floating=False):
    # A conservative static work estimate, not a latency guarantee. Avoid
    # trading tiny fused reductions for a sort plus extra kernel launches.
    if type(reference_count) is int:
        if reference_count < 32:
            return False
        search_levels = reference_count.bit_length()
        reference_range = True
        minimum_comparison_work = 262144
    else:
        # Bound symbolic reference lengths without reading their tracing hint.
        # Within this range, 17 is an upper bound on bit_length(reference_count).
        # Outside it, retain the original graph rather than specialize the size.
        search_levels = 17
        reference_range = (reference_count >= 32) & (reference_count <= 65536)
        # Full-path A/B showed extra sort/launch cost outweighing the saving at
        # 8192*128 comparisons, while 32768*128 retained a saving. Use the latter
        # as a conservative dynamic floor, not a universal break-even point.
        minimum_comparison_work = 4194304
    if floating:
        # Float normalization needs an extra reduction and more launches.
        # Ascend950PR full-path A/B: 32768*128 saved device time but increased
        # wall time; 262144*128 and 32768*1024 saved both. Use 2**25 as an
        # initial conservative floor, not a universal break-even point.
        minimum_comparison_work = 33554432
        if not config.allow_searchsorted_lowering:
            # AI_CPU search scales with the number of values. At 378000*378
            # it was slower than the original float reduction; at 32768*1024
            # the larger reference dimension amortized this cost. Retain the
            # original graph for short references on the fallback path.
            reference_range = reference_range & (reference_count >= 1024)
    comparison_work = value_count * reference_count
    search_work = (value_count + reference_count) * search_levels
    # These cutoffs are initial heuristics, not measured break-even points.
    # Minimum sizes leave room to amortize sorting and extra kernel launches.
    # The 262144 (2**18) floor limits rewriting to larger comparison workloads;
    # it was not derived from hardware limits or calibrated by a shape sweep.
    # Equal comparison counts can have different sort/search costs depending
    # on their two dimensions. The 4x work margin allows for costs omitted by
    # this estimate, but is likewise uncalibrated and does not ensure a speedup.
    # Tune these values using full-path A/B timings across shapes and dtypes.
    # Backed symbolic lengths use range guards, not exact-size guards. The
    # symbolic reference bound is also an initial, uncalibrated policy limit.
    # Unguardable data-dependent lengths conservatively retain the original graph.
    return guard_or_false(
        reference_range
        & (value_count >= 256)
        & (comparison_work >= minimum_comparison_work)
        & (comparison_work >= 4 * search_work)
    )


def _comparison(node, reduction_axis):
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target not in (
        aten.ge.Tensor, aten.gt.Tensor, aten.le.Tensor, aten.lt.Tensor
    ):
        return None
    lhs, rhs = _arg(node, 0, "self"), _arg(node, 1, "other")
    value_axis = 1 - reduction_axis
    values = _broadcast_vector(lhs, value_axis)
    reference_values = _broadcast_vector(rhs, reduction_axis)
    target = node.target
    if values is None or reference_values is None:
        values = _broadcast_vector(rhs, value_axis)
        reference_values = _broadcast_vector(lhs, reduction_axis)
        target = {
            aten.ge.Tensor: aten.le.Tensor,
            aten.gt.Tensor: aten.lt.Tensor,
            aten.le.Tensor: aten.ge.Tensor,
            aten.lt.Tensor: aten.gt.Tensor,
        }[target]
    if values is None or reference_values is None:
        return None
    # Use insertion indices for reference <= value / reference < value;
    # use the complement for reference >= value / reference > value.
    right = target in (aten.ge.Tensor, aten.lt.Tensor)
    complement = target in (aten.le.Tensor, aten.lt.Tensor)
    return values, reference_values, right, complement


def _match(sum_node):
    if not _is_call(sum_node, aten.sum.dim_IntList):
        return None
    dims = _arg(sum_node, 1, "dim")
    if not isinstance(dims, (tuple, list)) or len(dims) != 1:
        return None
    if type(dims[0]) is not int or dims[0] not in (0, 1, -1, -2):
        return None
    reduction_axis = dims[0] % 2
    keepdim = _arg(sum_node, 2, "keepdim", False)
    if type(keepdim) is not bool:
        return None
    if sum_node.kwargs.get("dtype") not in (None, torch.int64):
        return None
    comparison = _arg(sum_node, 0, "self")
    match = _comparison(comparison, reduction_axis)
    if match is None or len(comparison.users) != 1:
        return None
    values, reference_values, right, complement = match
    value_meta, reference_meta = values.meta["val"], reference_values.meta["val"]
    result = sum_node.meta.get("val")
    if not isinstance(result, FakeTensor):
        return None
    # Both lengths may be symbolic. Empty static inputs retain ATen; symbolic
    # positivity and profitability are checked together by the range guard.
    if any(
        type(t.shape[0]) is int and t.shape[0] <= 0
        for t in (value_meta, reference_meta)
    ):
        return None
    if (
        value_meta.device.type != "npu"
        or reference_meta.device != value_meta.device
        or result.device != value_meta.device
        or value_meta.dtype not in (
            torch.int32, torch.int64, torch.float32, torch.float16, torch.bfloat16
        )
        or reference_meta.dtype != value_meta.dtype
        or result.dtype != torch.int64
        or value_meta.layout != torch.strided
        or reference_meta.layout != torch.strided
        or value_meta.fake_mode is not reference_meta.fake_mode
        or value_meta.fake_mode is not result.fake_mode
    ):
        return None
    expected_shape = (
        (reference_meta.numel(), value_meta.numel())
        if reduction_axis == 0
        else (value_meta.numel(), reference_meta.numel())
    )
    if not _same_shape(_shape(comparison), expected_shape):
        return None
    if not _profitable(
        value_meta.numel(), reference_meta.numel(), value_meta.dtype.is_floating_point
    ):
        return None
    return values, reference_values, right, complement, reduction_axis, keepdim


def _call_with_meta(graph, target, args, kwargs=None):
    """Evaluate new ATen nodes only on metadata, including symbolic sizes."""
    kwargs = kwargs or {}
    node = graph.call_function(target, args=args, kwargs=kwargs)
    meta_args = tuple(
        a.meta["val"] if isinstance(a, torch.fx.Node) else a for a in args
    )
    node.meta["val"] = target(*meta_args, **kwargs)
    return node


@register_custom_pass(PassType.POST, ignore_inference_check=True)
def comparison_count_to_searchsorted_pass(graph: torch.fx.Graph) -> None:
    """Count with insertion indices; preserve comparison direction and rank."""
    if config.enable_full_lowering_fallback.strip() == "allfallback":
        return
    changed = False
    for node in list(graph.nodes):
        match = _match(node)
        if match is None:
            continue
        values, reference_values, right, complement, reduction_axis, keepdim = match
        value_meta, reference_meta = values.meta["val"], reference_values.meta["val"]
        floating = value_meta.dtype.is_floating_point
        query_nan = valid_count = None
        if floating:
            with graph.inserting_before(node), value_meta.fake_mode:
                if (
                    value_meta.dtype in (torch.float16, torch.bfloat16)
                    and not config.allow_searchsorted_lowering
                ):
                    # ACLNN SearchSorted rejects BF16; its FP16 path was slower
                    # than FP32 in full-path A/B. Both types widen exactly to
                    # FP32. Preserve original tensor users and native dtypes.
                    values = _call_with_meta(
                        graph, torch.ops.prims.convert_element_type.default,
                        (values, torch.float32),
                    )
                    reference_values = _call_with_meta(
                        graph, torch.ops.prims.convert_element_type.default,
                        (reference_values, torch.float32),
                    )
                    value_meta = values.meta["val"]
                    reference_meta = reference_values.meta["val"]
                # Never search NaNs: replace reference NaNs with +inf so sorting
                # and binary search operate on an ordered sequence. Capping the
                # insertion index by the non-NaN count also handles real +inf
                # references, which compare equal to these sentinels.
                reference_nan = _call_with_meta(
                    graph, aten.isnan.default, (reference_values,)
                )
                nan_count = _call_with_meta(
                    graph, aten.sum.dim_IntList, (reference_nan, [0])
                )
                negative_nan_count = _call_with_meta(
                    graph, aten.neg.default, (nan_count,)
                )
                size = reference_meta.numel()
                if type(size) is not int:
                    size = _call_with_meta(
                        graph, aten.sym_size.int, (reference_values, 0)
                    )
                valid_count = _call_with_meta(
                    graph, aten.add.Scalar, (negative_nan_count, size)
                )
                infinity = _call_with_meta(
                    graph, aten.full.default, ([], float("inf")),
                    {"dtype": reference_meta.dtype, "device": reference_meta.device},
                )
                reference_values = _call_with_meta(
                    graph, aten.where.self, (reference_nan, infinity, reference_values)
                )
                query_nan = _call_with_meta(graph, aten.isnan.default, (values,))
                zeros = _call_with_meta(
                    graph, aten.full.default, ([], 0),
                    {"dtype": value_meta.dtype, "device": value_meta.device},
                )
                values = _call_with_meta(
                    graph, aten.where.self, (query_nan, zeros, values)
                )
        intermediate_meta = node.meta.copy()
        if keepdim:
            # Intermediate counts are vectors; only the final unsqueeze may
            # inherit the original rank-two tensor metadata.
            intermediate_meta.pop("tensor_meta", None)
        # Sort/search metadata evaluation never inspects real values.
        with value_meta.fake_mode:
            sorted_reference_meta = aten.sort.default(reference_meta, -1, False)
            count_meta = aten.searchsorted.Tensor(
                sorted_reference_meta[0], value_meta, out_int32=False, right=right
            )
            negative_meta = -count_meta if complement else None
            complement_meta = (
                reference_meta.numel() - count_meta if complement else count_meta
            )
        with graph.inserting_before(node):
            sorted_reference_pair = graph.call_function(
                aten.sort.default, args=(reference_values, -1, False)
            )
            sorted_reference_pair.meta["val"] = sorted_reference_meta
            sorted_reference_values = graph.call_function(
                operator.getitem, args=(sorted_reference_pair, 0)
            )
            sorted_reference_values.meta["val"] = sorted_reference_meta[0]
            count = graph.call_function(
                aten.searchsorted.Tensor,
                args=(sorted_reference_values, values),
                kwargs={"out_int32": False, "right": right},
            )
            count.meta = {**intermediate_meta, "val": count_meta}
            if floating:
                with value_meta.fake_mode:
                    count = _call_with_meta(
                        graph, aten.minimum.default, (count, valid_count)
                    )
            if complement:
                # This pass runs after decomposition: rsub.Scalar has a decomp
                # but no direct lowering, so emit lowerable pointwise ops.
                count = graph.call_function(aten.neg.default, args=(count,))
                count.meta = {**intermediate_meta, "val": negative_meta}
                reference_count = valid_count if floating else reference_meta.numel()
                if not floating and type(reference_count) is not int:
                    # A symbolic length must be represented by an FX node,
                    # not embedded as a tracing-time SymInt argument.
                    size = graph.call_function(
                        aten.sym_size.int, args=(reference_values, 0)
                    )
                    size.meta["val"] = reference_count
                    reference_count = size
                count = graph.call_function(
                    aten.add.Tensor if floating else aten.add.Scalar,
                    args=(count, reference_count),
                )
                count.meta = {**intermediate_meta, "val": complement_meta}
            if floating:
                with value_meta.fake_mode:
                    # NaN queries yield zero for all four ordered comparisons.
                    zeros = _call_with_meta(
                        graph, aten.full.default, ([], 0),
                        {"dtype": torch.int64, "device": value_meta.device},
                    )
                    count = _call_with_meta(
                        graph, aten.where.self, (query_nan, zeros, count)
                    )
            if keepdim:
                count = graph.call_function(
                    aten.unsqueeze.default, args=(count, reduction_axis)
                )
                count.meta = node.meta.copy()
        node.replace_all_uses_with(count)
        graph.erase_node(node)
        counters["inductor"]["npu_comparison_count_to_searchsorted"] += 1
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
