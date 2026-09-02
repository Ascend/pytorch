import contextlib
import functools
import math
import operator
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.fx
from torch.utils._ordered_set import OrderedSet

from ...config import is_ascend950, log
from ..utils.check_op_util import (
    _get_tensor_meta,
    check_act_op,
    check_cat_op,
    check_op_by_targets,
    check_squeeze_op,
    check_support_op,
    check_unsqueeze_op,
    check_view,
    check_where_op,
    get_cast_dtype,
    get_input_kw_node,
    get_input_node,
    get_node_dtype,
    is_cast_node,
    is_one_like,
    is_single_user,
    is_zero_like,
    match,
    normalize_dim,
    normalize_dtype,
    try_match,
)
from ..utils.fx_pass_level import PassType
from ..utils.get_binary_fold_result import (
    _fold_slice,
    _fold_slice_scatter,
    _get_fold_result,
    get_binary_fold_result,
    get_node_meta,
    get_node_shape,
    get_node_unique_id,
    get_pad_dim_and_size,
    get_slice_dim,
    has_storage_or_layout,
    propagate_fake_tensor,
)
from ..utils.symbolic_shape_util import (
    is_statically_one,
    materialize_shape,
    refresh_fake_meta,
    resolve_size_arg,
    resolve_size_list,
    shapes_statically_equal,
    statically_fits_int32,
    statically_known_eq,
    statically_known_geq,
    statically_known_leq,
)
from .register_custom_pass import register_custom_pass


torch.library.define(
    "npu_ext::masked_fill_inf", "(Tensor x, Tensor mask, float value) -> Tensor"
)


torch.library.define(
    "npu_ext::multi_slice_concat",
    "(Tensor[] srcs, Tensor[] masks, int[] src_idx, int[] offsets, int[] widths, "
    "int[] mask_idx) -> Tensor",
)


def _multi_slice_concat_ref(srcs, masks, src_idx, offsets, widths, mask_idx):
    n = len(offsets)
    if not (len(src_idx) == len(widths) == len(mask_idx) == n):
        raise RuntimeError(
            "multi_slice_concat: segment arrays disagree in length "
            f"src_idx={len(src_idx)} offsets={n} widths={len(widths)} "
            f"mask_idx={len(mask_idx)}"
        )
    if n == 0:
        raise RuntimeError("multi_slice_concat: needs at least one segment")
    if not srcs:
        raise RuntimeError("multi_slice_concat: source tensor list is empty")

    parts = []
    for i in range(n):
        si, off, width, mi = src_idx[i], offsets[i], widths[i], mask_idx[i]
        if not 0 <= si < len(srcs):
            raise RuntimeError(f"multi_slice_concat: segment {i} src_idx={si} out of range")
        src = srcs[si]
        if src.dim() != 2:
            raise RuntimeError(f"multi_slice_concat: segment {i} source is not 2D")
        if width <= 0 or off < 0:
            raise RuntimeError(
                f"multi_slice_concat: segment {i} has invalid offset={off} width={width}"
            )
        limit = src.shape[-1]
        if isinstance(limit, int) and off + width > limit:
            raise RuntimeError(
                f"multi_slice_concat: segment {i} spans [{off}, {off + width}) "
                f"beyond last dim {limit}"
            )
        part = torch.ops.aten.slice.Tensor(src, -1, off, off + width)
        if mi >= 0:
            if mi >= len(masks):
                raise RuntimeError(f"multi_slice_concat: segment {i} mask_idx={mi} out of range")
            mask = masks[mi]
            # A row mask must be [rows, 1] to broadcast across columns. A 1D mask would
            # broadcast along columns instead, inverting the meaning, so reject it.
            if mask.dim() != 2 or (
                isinstance(mask.shape[1], int) and mask.shape[1] != 1
            ):
                raise RuntimeError(
                    f"multi_slice_concat: segment {i} mask shape {tuple(mask.shape)} "
                    "is not [rows, 1]"
                )
            part = torch.ops.aten.where.self(
                mask, torch.ops.aten.zeros_like.default(part), part
            )
        parts.append(part)
    return torch.ops.aten.cat.default(parts, -1)


# CompositeExplicitAutograd also covers the Meta key, and the reference is built purely
# from aten ops, so fake propagation follows without a separate fake kernel.
torch.library.impl(
    "npu_ext::multi_slice_concat", "CompositeExplicitAutograd"
)(_multi_slice_concat_ref)

MULTI_SLICE_CONCAT_TARGET = torch.ops.npu_ext.multi_slice_concat.default


@register_custom_pass(PassType.PRE)
def cat_slice_cat_fold_pass(graph: torch.fx.Graph) -> None:
    """Fold the redundant cat -> slice -> cat pattern: when the later cat takes contiguous
    slices of the earlier cat that fully cover it, reuse the earlier cat's result."""
    changed = False
    for node in reversed(list(graph.nodes)):
        if node.op != "call_function" or node.target not in (torch.cat, torch.concat):
            continue
        cat2_node = node
        cat2_inputs = cat2_node.args[0]
        cat2_dim = cat2_node.kwargs.get("dim", -1)
        cat2_shape = get_node_shape(cat2_node, allow_symbolic=True)
        if not cat2_shape:
            continue
        cat2_rank = len(cat2_shape)
        cat2_dim = cat2_dim + cat2_rank if cat2_dim == -1 else cat2_dim
        all_slices = True
        slice_inputs = []
        slice_ranges = []
        for inp in cat2_inputs:
            if inp.op != "call_function" or inp.target != operator.getitem:
                all_slices = False
                break
            if len(inp.args) < 2 or not isinstance(inp.args[1], tuple):
                all_slices = False
                break
            slice_input = inp.args[0]
            slice_args = inp.args[1]
            slice_dim = get_slice_dim(slice_args, cat2_dim)
            if slice_dim is None or slice_dim != cat2_dim:
                all_slices = False
                break
            slice_ranges.append(slice_args[cat2_dim])
            slice_inputs.append(slice_input)
        if not all_slices:
            continue
        cat1_node = slice_inputs[0]
        if not all(s == cat1_node for s in slice_inputs):
            continue
        if cat1_node.op != "call_function" or cat1_node.target not in (
            torch.cat,
            torch.concat,
        ):
            continue
        cat1_inputs = cat1_node.args[0]
        cat1_dim = cat1_node.kwargs.get("dim", -1)
        cat1_shape = get_node_shape(cat1_node, allow_symbolic=True)
        if not cat1_shape:
            continue
        cat1_rank = len(cat1_shape)
        cat1_dim = cat1_dim + cat1_rank if cat1_dim == -1 else cat1_dim
        if cat1_dim != cat2_dim or not shapes_statically_equal(cat1_shape, cat2_shape):
            continue
        # Normalize each slice's (start, stop), allowing symbolic bounds (step must be 1).
        resolved_ranges = []
        valid_ranges = True
        for sl in slice_ranges:
            if sl.step not in (1, None):
                valid_ranges = False
                break
            start = resolve_size_arg(0 if sl.start is None else sl.start)
            stop = resolve_size_arg(sl.stop) if sl.stop is not None else None
            if start is None or stop is None:
                valid_ranges = False
                break
            resolved_ranges.append((start, stop))
        if not valid_ranges or len(resolved_ranges) != len(cat1_inputs):
            continue
        # Chained coverage proof: start_0==0, start_i==stop_{i-1}, last stop covers the full dim length.
        ranges_match = True
        expected_start = 0
        for start, stop in resolved_ranges:
            if not statically_known_eq(start, expected_start):
                ranges_match = False
                break
            expected_start = stop
        if ranges_match and not statically_known_eq(
            expected_start, cat2_shape[cat2_dim]
        ):
            ranges_match = False

        if not ranges_match:
            continue
        with graph.inserting_before(cat2_node):
            cat2_node.replace_all_uses_with(cat1_node)
        graph.erase_node(cat2_node)
        for slice_node in cat2_inputs:
            graph.erase_node(slice_node)
        changed = True
    eliminate_dead_code(graph, changed, cat_slice_cat_fold_pass.__name__, False)


@register_custom_pass(PassType.PRE)
def pad_slice_fold(graph: torch.fx.Graph) -> None:
    """Fold the pad -> slice pattern: when the slice range lies inside the valid data
    region ahead of the pad, slice the original input directly and drop the pad node."""
    # padding -> slice
    changed = False
    for node in reversed(list(graph.nodes)):
        # check whether this is a linear node
        if node.op != "call_function" or node.target != torch._C._nn.pad:
            continue
        # get the pad node's inputs and arguments
        input_tensor = node.args[0]
        pad = node.args[1]
        input_shape = get_node_shape(input_tensor, allow_symbolic=True)
        if input_shape is None:
            continue
        pad_dim, _ = get_pad_dim_and_size(pad, input_shape)
        if pad_dim is None:
            continue
        # find the consumers of the pad node
        # check every downstream slice node
        all_slices_valid = True
        slice_nodes = []
        for user in list(node.users):
            if user.op != "call_function" or user.target != operator.getitem:
                all_slices_valid = False
                break
            # take out the index tuple
            idx = user.args[1]
            if not isinstance(idx, (tuple, list)) or len(idx) <= pad_dim:
                all_slices_valid = False
                break
            start = idx[pad_dim].start
            end = idx[pad_dim].stop
            step = idx[pad_dim].step
            slice_start = 0 if start is None else resolve_size_arg(start)
            slice_end = None if end is None else resolve_size_arg(end)
            slice_step = step if isinstance(step, int) else 1
            # The slice upper bound must be provably within the valid pre-pad data region (not touching padding).
            is_valid_prefix = (
                slice_start is not None
                and slice_end is not None
                and slice_step in (1, None)
                and statically_known_leq(slice_end, input_shape[pad_dim])
                and statically_known_leq(slice_start, slice_end)
            )
            if not is_valid_prefix:
                all_slices_valid = False
                break
            slice_nodes.append((user, (input_tensor, idx)))

        # if every slice node qualifies, replace pad + slice with a direct slice
        if all_slices_valid and slice_nodes:
            for user, new_args in slice_nodes:
                user.args = new_args
            graph.erase_node(node)  # erase the pad node
            changed = True
    eliminate_dead_code(graph, changed, pad_slice_fold.__name__, False)


@register_custom_pass(PassType.POST)
def fold_four_op_pass(graph: torch.fx.Graph) -> None:
    """Remove identity arithmetic such as x+0, x-0, 0-x, x*1 and x/1 by replacing
    the whole binary op node with the non-zero / non-one operand."""
    changed = False
    add_ops = (torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
    sub_ops = (torch.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
    rsub_ops = (torch.rsub, torch.ops.aten.rsub.Tensor, torch.ops.aten.rsub.Scalar)
    mul_ops = (torch.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
    div_ops = (torch.div, torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar)
    changed = True
    total_changed = False
    while changed:
        changed = False
        for node in reversed(list(graph.nodes)):
            if node.op != "call_function" or node.target not in (
                add_ops + sub_ops + rsub_ops + mul_ops + div_ops
            ):
                continue
            if len(node.args) < 2:
                continue
            inp0 = node.args[0]
            inp1 = node.args[1]
            target_val = None
            is_match = False
            if check_op_by_targets(node, add_ops):
                is_match, target_val = try_match(inp0, inp1, is_zero_like)
            elif check_op_by_targets(node, sub_ops):
                is_match, target_val = try_match(inp0, inp1, is_zero_like, "right")
            elif check_op_by_targets(node, rsub_ops):
                is_match, target_val = try_match(inp0, inp1, is_zero_like, "left")
            elif check_op_by_targets(node, div_ops):
                is_match, target_val = try_match(inp0, inp1, is_one_like, "right")
            elif check_op_by_targets(node, mul_ops):
                is_match, target_val = try_match(inp0, inp1, is_one_like)

            if is_match:
                with graph.inserting_before(node):
                    fold_res = get_binary_fold_result(graph, target_val, node.meta)
                if fold_res is not None:
                    node.replace_all_uses_with(fold_res)
                    graph.erase_node(node)
                    changed = True
                    total_changed = True
    if total_changed:
        eliminate_dead_code(graph, total_changed, fold_four_op_pass.__name__)


@register_custom_pass(PassType.POST)
def fold_cast(graph: torch.fx.Graph) -> None:
    """Remove identity casts: when the cast target dtype equals the input dtype,
    replace the cast node with its input."""
    changed = False

    for node in list(graph.nodes):
        if not is_cast_node(node):
            continue

        src_cast = node
        if len(src_cast.args) == 0 or not isinstance(src_cast.args[0], torch.fx.Node):
            continue
        src_input = src_cast.args[0]
        src_input_dtype = normalize_dtype(get_node_dtype(src_input))
        cur_cast_dtype = normalize_dtype(get_cast_dtype(src_cast))

        if src_input_dtype is None or cur_cast_dtype is None:
            continue
        if src_input_dtype == cur_cast_dtype:
            with graph.inserting_before(src_cast):
                src_cast.replace_all_uses_with(src_input)
                propagate_fake_tensor(src_input, src_cast, lambda x: x)
            graph.erase_node(src_cast)
            changed = True
    eliminate_dead_code(graph, changed, fold_cast.__name__)


@register_custom_pass(PassType.POST)
def fold_cat(graph: torch.fx.Graph) -> None:
    """Merge nested cats: when a cat input is itself a single-use cat on the same dim,
    flatten the inner cat's inputs into the outer one to save a concatenation."""
    changed = False
    flag = True
    while flag:
        flag = False
        for node in list(graph.nodes):
            is_cat, cat_axis = check_cat_op(node)
            if not is_cat:
                continue
            node_shape = get_node_shape(node, allow_symbolic=True)
            if not node_shape:
                continue
            if cat_axis == len(node_shape) - 1:
                cat_axis = -1
            cat_input = []
            foldable = False
            for inp in node.args[0]:
                is_input_cat, input_cat_axis = check_cat_op(inp)
                if is_input_cat:
                    if len(inp.users) == 1:
                        inp_shape = get_node_shape(inp, allow_symbolic=True)
                        effective_input_axis = input_cat_axis
                        if inp_shape and input_cat_axis == len(inp_shape) - 1:
                            effective_input_axis = -1

                        if cat_axis == effective_input_axis:
                            cat_input += inp.args[0]
                            foldable = True
                        else:
                            cat_input.append(inp)
                    else:
                        cat_input.append(inp)
                else:
                    cat_input.append(inp)
            if foldable:
                with graph.inserting_before(node):
                    concat_node = graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(cat_input, cat_axis),
                        name=node.name + "_1",
                    )
                    propagate_fake_tensor(
                        concat_node,
                        cat_input,
                        lambda fake: concat_node.target(fake, cat_axis),
                    )
                node.replace_all_uses_with(concat_node)
                graph.erase_node(node)
                changed = True
                flag = True
    eliminate_dead_code(graph, changed, fold_cat.__name__)


@register_custom_pass(PassType.POST)
def fold_clone(graph: torch.fx.Graph) -> None:
    """Remove clones that keep memory_format and are not graph outputs: when the clone
    does not affect storage semantics, replace it with its input."""
    changed = False
    output_node: torch.fx.Node = list(graph.nodes)[-1]
    if output_node.op != "output":
        return
    output_storages = OrderedSet()
    for n in output_node.all_input_nodes:
        identity = get_node_unique_id(n)
        if identity is not None:
            output_storages.add(identity)
    if not output_storages:
        return

    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.clone.default
        and has_storage_or_layout(node)
        and get_node_unique_id(node) not in output_storages
    ]
    for clone in candidates:
        inp = clone.args[0]
        if "tensor_meta" not in inp.meta:
            continue
        org_memoryformat = inp.meta["tensor_meta"].memory_format
        target_memoryformat = clone.kwargs.get("memory_format", org_memoryformat)
        if org_memoryformat == target_memoryformat:
            clone.replace_all_uses_with(inp)
            propagate_fake_tensor(inp, clone, lambda x: x)
            graph.erase_node(clone)
            changed = True
    eliminate_dead_code(graph, changed, fold_clone.__name__)


@register_custom_pass(PassType.POST)
def fold_detach(graph: torch.fx.Graph) -> None:
    """Remove detach nodes in inference graphs: detach does not change forward values,
    so it can be replaced by its input."""
    changed = False
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.detach.default
    ]
    for detach in candidates:
        inp = detach.args[0]
        detach.replace_all_uses_with(inp)
        propagate_fake_tensor(inp, detach, lambda x: x)
        graph.erase_node(detach)
        changed = True
    eliminate_dead_code(graph, changed, fold_detach.__name__)


@register_custom_pass(PassType.POST)
def fold_expand(graph: torch.fx.Graph) -> None:
    """Remove identity expands: when the target shape matches the input shape (-1 counts
    as equal), replace the expand node with its input."""
    changed = False
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.expand.default
    ]

    def _same_shape(org_shape, target_shape) -> bool:
        """Tell whether two shapes are equivalent (-1 in the target matches the original dim)."""
        if len(org_shape) != len(target_shape):
            return False
        for os, ts in zip(org_shape, target_shape):
            if isinstance(ts, int) and ts == -1:
                continue
            if not statically_known_eq(os, ts):
                return False
        return True

    for expand in candidates:
        inp = expand.args[0]
        target_shape = expand.args[1]
        if not isinstance(target_shape, (list, tuple)):
            continue
        target_shape = resolve_size_list(target_shape)
        if target_shape is None:
            continue
        inp_shape = get_node_shape(inp, allow_symbolic=True)
        if inp_shape is None:
            continue
        org_shape = list(inp_shape)
        if _same_shape(org_shape, target_shape):
            expand.replace_all_uses_with(inp)
            propagate_fake_tensor(inp, expand, lambda x: x)
            graph.erase_node(expand)
            changed = True
    eliminate_dead_code(graph, changed, fold_expand.__name__)


@register_custom_pass(PassType.POST)
def fold_reduce(graph: torch.fx.Graph) -> None:
    """Remove reduces (such as sum) over size-1 dims: they do not change the values,
    so they can be replaced by the equivalent view op."""
    changed = False
    reduce_tup = (torch.ops.aten.sum.dim_IntList,)
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target in reduce_tup
    ]

    for reduce in reversed(candidates):
        inp = get_input_node(reduce, 0)
        shape = get_node_shape(inp, allow_symbolic=True)
        if shape is None:
            continue
        dims = get_input_kw_node(reduce, "dim") or list(range(len(shape)))
        if not isinstance(dims, list):
            dims = [dims]
        keep_dim = get_input_kw_node(reduce, "keepdim") or False
        if all(is_statically_one(shape[dim]) for dim in dims):
            with graph.inserting_before(reduce):
                fold_res = _get_fold_result(graph, inp, dims, keep_dim)
            if fold_res:
                reduce.replace_all_uses_with(fold_res)
                graph.erase_node(reduce)
                changed = True
    eliminate_dead_code(graph, changed, fold_reduce.__name__)


@register_custom_pass(PassType.POST)
def fold_sink_view(graph: torch.fx.Graph) -> None:
    """Sink a view past the following activation / pointwise op: compute on the original
    shape first and view afterwards, which helps fusion without changing values."""
    changed = False
    for node in reversed(graph.nodes):
        if not check_view(node):
            continue
        if len(node.users) != 1:
            continue
        view_shape = get_node_shape(node, allow_symbolic=True)
        if view_shape is None:
            continue
        user = next(iter(node.users))
        if check_act_op(user)[0]:
            with graph.inserting_before(user):
                new_act = graph.create_node(
                    op="call_function",
                    target=user.target,
                    args=(node.args[0],),
                    name=user.name + "_replacement",
                )
                propagate_fake_tensor(
                    new_act, node.args[0], lambda fake: user.target(fake)
                )
                new_act_view = graph.create_node(
                    op="call_function",
                    target=node.target,
                    args=(new_act, node.args[1]),
                    name=node.name + "_replacement",
                )
                propagate_fake_tensor(
                    new_act_view, new_act, lambda fake: node.target(fake, view_shape)
                )
            user.replace_all_uses_with(new_act_view)
            graph.erase_node(user)
            changed = True
        elif check_support_op(user):
            if user.args[0] is node:
                other_node = user.args[1]
            else:
                other_node = user.args[0]
            if isinstance(other_node, (float, int, bool)):
                other_shape = []
                other_val = other_node
            else:
                other_shape = get_node_shape(other_node, allow_symbolic=True)
                other_val = get_node_meta(other_node)
            result_shape = get_node_shape(user, allow_symbolic=True)
            orig_shape = get_node_shape(node.args[0], allow_symbolic=True)
            if (
                other_shape is not None
                and result_shape is not None
                and view_shape is not None
                and orig_shape is not None
            ):
                no_broadcast_dims = min(len(other_shape), len(orig_shape))
                if shapes_statically_equal(result_shape, view_shape) and (
                    len(other_shape) == 0
                    or shapes_statically_equal(
                        orig_shape[-no_broadcast_dims:],
                        view_shape[-no_broadcast_dims:],
                    )
                ):
                    with graph.inserting_before(user):
                        new_args = list(user.args)
                        for x, arg in enumerate(new_args):
                            if arg is node:
                                new_args[x] = node.args[0]
                                view_index = x
                        new_add = graph.create_node(
                            op="call_function",
                            target=user.target,
                            args=tuple(new_args),
                            name=user.name + "_replacement",
                        )
                        if view_index == 0:
                            propagate_fake_tensor(
                                new_add,
                                node.args[0],
                                lambda fake: user.target(fake, other_val),
                            )
                        else:
                            propagate_fake_tensor(
                                new_add,
                                node.args[0],
                                lambda fake: user.target(other_val, fake),
                            )
                        new_add_view = graph.create_node(
                            op="call_function",
                            target=node.target,
                            args=(new_add, node.args[1]),
                            name=node.name + "_replacement",
                        )

                        propagate_fake_tensor(
                            new_add_view,
                            new_add,
                            lambda fake: node.target(fake, view_shape),
                        )
                    user.replace_all_uses_with(new_add_view)
                    graph.erase_node(user)
                    changed = True
    eliminate_dead_code(graph, changed, fold_sink_view.__name__)


@register_custom_pass(PassType.POST)
def fold_slice(graph: torch.fx.Graph) -> None:
    """Fold no-op slice / slice_scatter: when the slice range covers the full range,
    replace it with the input to drop the redundant slice."""
    changed = False
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if node.target == torch.ops.aten.slice.Tensor:
            if _fold_slice(node, graph):
                changed = True
                log.info("FoldSliceLike: Folded slice node %s", node.name)
        elif node.target == torch.ops.aten.slice_scatter.default:
            if _fold_slice_scatter(node, graph):
                log.info("FoldSliceLike: Folded slice_scatter node %s", node.name)
                changed = True
    eliminate_dead_code(graph, changed, fold_slice.__name__)


@register_custom_pass(PassType.POST)
def fold_squeeze(graph: torch.fx.Graph) -> None:
    """Merge adjacent squeeze/unsqueeze: handles chained squeeze->squeeze and the
    mutually inverse squeeze->unsqueeze, both redundant reshapes."""
    changed = False
    for node in reversed(graph.nodes):
        if not check_squeeze_op(node):
            continue
        prev = node.args[0]
        if len(prev.users) > 1:
            continue
        # case1: squeeze -> squeeze
        if check_squeeze_op(prev):
            if len(node.args) == 1:
                node.replace_input_with(prev, prev.args[0])
                changed = True
            elif len(prev.args) == 1:
                node.replace_all_uses_with(prev)
                propagate_fake_tensor(prev, node, lambda x: x)
                graph.erase_node(node)
                changed = True
        # case2: squeeze -> unsqueeze
        elif check_unsqueeze_op(prev):
            if len(node.args) == 1:
                node.replace_input_with(prev, prev.args[0])
                changed = True
            elif match(prev.args[1], node.args[1]):
                node.replace_all_uses_with(prev.args[0])
                propagate_fake_tensor(prev.args[0], node, lambda x: x)
                changed = True
    eliminate_dead_code(graph, changed, fold_squeeze.__name__)


@register_custom_pass(PassType.POST)
def fold_to_copy(graph: torch.fx.Graph) -> None:
    """Remove side-effect-free _to_copy: when dtype/device/memory_format are unchanged
    and the result is not a graph output, replace it with its input."""
    changed = False
    output_node: torch.fx.Node = list(graph.nodes)[-1]
    if output_node.op != "output":
        return
    output_storages = OrderedSet()
    for n in output_node.all_input_nodes:
        identity = get_node_unique_id(n)
        if identity is not None:
            output_storages.add(identity)
    if not output_storages:
        return
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten._to_copy.default
        and has_storage_or_layout(node)
        and get_node_unique_id(node) not in output_storages
    ]

    def _useless_to_copy(copy: torch.fx.Node) -> bool:
        """Tell whether a _to_copy node is a no-op copy: every observable attribute
        (dtype, device, layout, memory_format and so on) matches the input."""
        inp = copy.args[0]
        copy_dtype = copy.kwargs.get("dtype", None)
        copy_meta = get_node_meta(copy)
        in_meta = get_node_meta(inp)
        if copy_meta is None or in_meta is None:
            return False
        if copy_dtype is not None and copy_dtype != in_meta.dtype:
            return False
        if in_meta.dtype != copy_meta.dtype:
            return False
        if "layout" in copy.kwargs:
            return False

        if hasattr(copy_meta, "device") and hasattr(in_meta, "device"):
            if in_meta.device != copy_meta.device:
                return False

        if "pin_memory" in copy.kwargs or "non_blocking" in copy.kwargs:
            return False
        if "memory_format" in copy.kwargs:
            return (
                "tensor_meta" in inp.meta
                and "tensor_meta" in copy.meta
                and inp.meta["tensor_meta"].memory_format
                == copy.meta["tensor_meta"].memory_format
            )
        return True

    for _to_copy in candidates:
        if _useless_to_copy(_to_copy):
            _to_copy.replace_all_uses_with(_to_copy.args[0])
            propagate_fake_tensor(_to_copy.args[0], _to_copy, lambda x: x)
            graph.erase_node(_to_copy)
            changed = True
    eliminate_dead_code(graph, changed, fold_to_copy.__name__)


@register_custom_pass(PassType.POST)
def view_fold_pass(graph) -> None:
    """Fold chained view-like ops: collapse a view/reshape/squeeze/unsqueeze chain into a
    single reshape, and drop identity views whose target shape equals the input shape."""
    changed = False
    view_tup = (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    )
    _view_like_ops = (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
    )
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target in view_tup
    ]
    for view in candidates:
        inp = view.args[0]
        if (
            isinstance(inp, torch.fx.Node)
            and inp.op == "call_function"
            and inp.target in _view_like_ops
        ):
            view.replace_input_with(inp, inp.args[0])
            changed = True
        else:
            target_shape = view.args[1]
            if not isinstance(target_shape, (list, tuple)):
                continue
            target_shape = resolve_size_list(target_shape)
            if target_shape is None:
                continue
            inp_shape = get_node_shape(inp, allow_symbolic=True)
            if inp_shape is not None:
                if shapes_statically_equal(target_shape, list(inp_shape)):
                    view.replace_all_uses_with(inp)
                    propagate_fake_tensor(inp, view, lambda x: x)
                    graph.erase_node(view)
                    changed = True
    eliminate_dead_code(graph, changed, view_fold_pass.__name__)


@register_custom_pass(PassType.POST)
def fold_where(graph: torch.fx.Graph) -> None:
    """Fold a where whose branches are identical: when the true / false branches hold the
    same value (or are both all-0 / all-1), replace the where with either branch."""
    changed = False

    for where in reversed(graph.nodes):
        if not check_where_op(where):
            continue
        inp = where.args[1]
        other = where.args[2]
        if (
            (inp == other)
            or (is_one_like(inp) and is_one_like(other))
            or (is_zero_like(inp) and is_zero_like(other))
        ):
            with graph.inserting_before(where):
                res = get_binary_fold_result(graph, inp, where.meta)

            if res is not None:
                where.replace_all_uses_with(res)
                graph.erase_node(where)
                changed = True
    eliminate_dead_code(graph, changed, fold_where.__name__)


@register_custom_pass(PassType.POST)
def fold_redundant_ops(graph: torch.fx.Graph):
    """Remove a redundant view->squeeze pair: when the squeeze output shape/dtype after
    the view matches the view input exactly, replace the whole chain with that input."""
    changed = False
    while True:
        any_removed = False
        nodes = list(graph.nodes)
        for node in nodes:
            if node.op != "call_function":
                continue
            if node.target not in (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
            ):
                continue

            view_node = node
            if not view_node.args:
                continue
            first_arg = view_node.args[0]
            if not isinstance(first_arg, torch.fx.Node):
                continue
            users = list(view_node.users)
            for user in users:
                if (
                    user.op != "call_function"
                    or user.target != torch.ops.aten.squeeze.dim
                ):
                    continue
                squeeze_node = user
                if not squeeze_node.args:
                    continue
                if squeeze_node.args[0] is not view_node:
                    continue
                in_meta = _get_tensor_meta(first_arg)
                squeeze_out_meta = _get_tensor_meta(squeeze_node)
                in_shape = get_node_shape(first_arg, allow_symbolic=True)
                squeeze_out_shape = get_node_shape(squeeze_node, allow_symbolic=True)
                if in_meta is None or squeeze_out_meta is None:
                    continue
                if in_shape is None or squeeze_out_shape is None:
                    continue
                if not shapes_statically_equal(in_shape, squeeze_out_shape):
                    continue
                if in_meta.dtype != squeeze_out_meta.dtype:
                    continue
                squeeze_node.replace_all_uses_with(first_arg)
                propagate_fake_tensor(first_arg, squeeze_node, lambda x: x)
                graph.erase_node(squeeze_node)
                if not list(view_node.users):
                    graph.erase_node(view_node)

                any_removed = True
                changed = True
                break

            if any_removed:
                break

        if not any_removed:
            break
    eliminate_dead_code(graph, changed, fold_redundant_ops.__name__)


@register_custom_pass(PassType.PRE)
def dtype_optimal_pass(graph: torch.fx.Graph) -> None:
    """Narrow unnecessary int64 to int32: if the values of torch.arange or to(int64)
    fit safely in int32, downgrade the dtype to cut memory traffic and compute cost."""
    cast_dtype_limit = [torch.float32, torch.int32, torch.bool, torch.int16, torch.int8]
    changed = False
    for node in list(graph.nodes):  # use list to avoid mutating while iterating
        if (
            node.op == "call_function"
            and node.target == torch.arange
            and node.kwargs.get("dtype", None) == torch.int64
        ):
            # step 1: extract start/end/step dynamically (handles different args lengths)
            args_len = len(node.args)
            start = 0
            end = None
            step = 1
            if args_len == 1:
                end = node.args[0]  # arange(end)
            elif args_len == 2:
                start = node.args[0]
                end = node.args[1]  # arange(start, end)
            elif args_len >= 3:
                start = node.args[0]
                end = node.args[1]
                step = node.args[2]  # arange(start, end, step)
            # merge kwargs overrides (e.g. a user-specified kwargs['start'])
            start = node.kwargs.get("start", start)
            end = node.kwargs.get("end", end)
            step = node.kwargs.get("step", step)
            # if end is None, treat it as unbounded and skip (rare, but safe)
            if end is None:
                continue
            # Normalize start/end/step (symbolic allowed); elements always lie in
            # [start, end), so downgrade is safe once both bounds provably fit int32; step must be a nonzero int.
            r_start = resolve_size_arg(start)
            r_end = resolve_size_arg(end)
            r_step = resolve_size_arg(step)
            if r_start is None or r_end is None or r_step is None:
                continue
            if not isinstance(r_step, int) or r_step == 0:
                continue
            if statically_fits_int32(r_start, r_end):
                node.kwargs = {**node.kwargs, "dtype": torch.int32}
                changed = True
        if node.op == "call_method":
            input_node = node.args[0]
            input_fake = (
                input_node.meta.get("example_value", None)
                if hasattr(input_node, "meta")
                else None
            )
            input_dtype = input_fake.dtype if input_fake is not None else None
            target_dtype = (
                node.args[1] if len(node.args) > 1 else node.kwargs.get("dtype", None)
            )
            if (
                input_dtype in cast_dtype_limit
                and node.target == "to"
                and target_dtype == torch.int64
            ):
                if len(node.args) > 1:
                    node.args = (node.args[0], torch.int32)  # update the positional dtype
                else:
                    node.kwargs = {
                        **node.kwargs,
                        "dtype": torch.int32,
                    }  # update the kwargs dtype
                changed = True
    eliminate_dead_code(graph, changed, dtype_optimal_pass.__name__, False)


@register_custom_pass(PassType.PRE)
def fusion_attention_v3_pass(graph: torch.fx.Graph) -> None:
    """Upgrade npu_fusion_attention to v3: replace the node with the equivalent v3 op,
    keeping all arguments and metadata, to enable the faster implementation."""
    changed = False
    for node in list(graph.nodes):  # use list to avoid mutating the graph while iterating
        if (
            node.op == "call_function"
            and node.target == torch.ops.npu.npu_fusion_attention.default
        ):
            # create a new node calling the v3 version
            with graph.inserting_before(node):
                new_node = graph.call_function(
                    torch.ops.npu.npu_fusion_attention_v3.default,
                    args=node.args,
                    kwargs=node.kwargs,
                )
                new_node.meta.update(node.meta)
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            changed = True
    eliminate_dead_code(graph, changed, fusion_attention_v3_pass.__name__, False)


@register_custom_pass(PassType.POST)
def cat_to_view_pass(graph: torch.fx.Graph) -> None:
    """Turn a cat of slices from one parent tensor into a view or roll: when the slices
    fully cover a dim the cat is an identity view or a cyclic shift, so no data moves."""
    target_cat = torch.ops.aten.cat.default
    target_slice = torch.ops.aten.slice.Tensor
    changed = False

    for cat in list(graph.nodes):
        if cat.op != "call_function" or cat.target != target_cat:
            continue
        if not cat.args:
            continue
        cat_inputs = cat.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) < 2:
            continue
        cat_shape = get_node_shape(cat, allow_symbolic=True)
        if cat_shape is None:
            continue
        rank = len(cat_shape)
        cat_dim_raw = cat.args[1] if len(cat.args) > 1 else cat.kwargs.get("dim", 0)
        if not isinstance(cat_dim_raw, int):
            continue
        cat_dim = cat_dim_raw + rank if cat_dim_raw < 0 else cat_dim_raw

        parent = None
        intervals = []
        valid = True
        all_static = True
        for inp in cat_inputs:
            if not (
                isinstance(inp, torch.fx.Node)
                and inp.op == "call_function"
                and inp.target == target_slice
                and len(inp.args) >= 2
            ):
                valid = False
                break
            p = inp.args[0]
            sl_dim_raw = inp.args[1]
            if not isinstance(sl_dim_raw, int):
                valid = False
                break
            sl_dim = sl_dim_raw + rank if sl_dim_raw < 0 else sl_dim_raw
            if sl_dim != cat_dim:
                valid = False
                break
            sl_step = inp.args[4] if len(inp.args) > 4 else 1
            if sl_step not in (1, None):
                valid = False
                break
            sl_start = inp.args[2] if len(inp.args) > 2 else 0
            sl_end = inp.args[3] if len(inp.args) > 3 else None
            r_start = resolve_size_arg(0 if sl_start is None else sl_start)
            r_end = None if sl_end is None else resolve_size_arg(sl_end)
            if r_start is None or (sl_end is not None and r_end is None):
                valid = False
                break
            if not isinstance(r_start, int) or (
                r_end is not None and not isinstance(r_end, int)
            ):
                all_static = False
            if parent is None:
                parent = p
            elif parent is not p:
                valid = False
                break
            intervals.append((r_start, r_end))

        if not valid or parent is None:
            continue
        parent_shape = get_node_shape(parent, allow_symbolic=True)
        if parent_shape is None or len(parent_shape) != rank:
            continue
        if not shapes_statically_equal(parent_shape, cat_shape):
            continue
        dim_size = parent_shape[cat_dim]

        if not (all_static and isinstance(dim_size, int)):
            # Symbolic path: negative indices cannot be reliably normalized under
            # symbols; only fold when the slices, in cat input order, contiguously
            # cover the full dim length from 0 (identity view); skip rotation cases.
            expected = 0
            ok_sym = True
            for s, e in intervals:
                e_eff = dim_size if e is None else e
                if not statically_known_geq(s, 0) or not statically_known_eq(
                    s, expected
                ):
                    ok_sym = False
                    break
                expected = e_eff
            if ok_sym and statically_known_eq(expected, dim_size):
                cat.replace_all_uses_with(parent)
                changed = True
                log.info(
                    "cat_to_view_pass: collapsed cat(%d slices, dim=%d) of %s "
                    "-> identity view (dynamic full cover)",
                    len(cat_inputs),
                    cat_dim,
                    parent.name,
                )
            continue

        dim_size = int(dim_size)

        normalised = []
        ok = True
        for s, e in intervals:
            s_norm = s if s >= 0 else s + dim_size
            e_norm = dim_size if e is None else (e if e >= 0 else e + dim_size)
            e_norm = min(e_norm, dim_size)
            if s_norm < 0 or e_norm <= s_norm:
                ok = False
                break
            normalised.append((s_norm, e_norm))
        if not ok:
            continue

        sorted_intervals = sorted(normalised, key=lambda se: se[0])
        expected = 0
        full_cover = True
        for s, e in sorted_intervals:
            if s != expected:
                full_cover = False
                break
            expected = e
        if not full_cover or expected != dim_size:
            continue

        if normalised == sorted_intervals:
            cat.replace_all_uses_with(parent)
            changed = True
            log.info(
                "cat_to_view_pass: collapsed cat(%d slices, dim=%d) of %s "
                "-> identity view (full cover [0, %d))",
                len(cat_inputs),
                cat_dim,
                parent.name,
                dim_size,
            )
            continue

        rotation = None
        n_blocks = len(sorted_intervals)
        for i in range(1, n_blocks):
            if normalised == sorted_intervals[i:] + sorted_intervals[:i]:
                rotation = i
                break
        if rotation is None:
            continue

        shift = -normalised[0][0]

        parent_fake = parent.meta.get("val")
        fake_mode = (
            getattr(parent_fake, "fake_mode", None) if parent_fake is not None else None
        )
        roll_fake = None
        if fake_mode is not None and parent_fake is not None:
            try:
                with fake_mode:
                    roll_fake = torch.ops.aten.roll.default(
                        parent_fake,
                        [shift],
                        [cat_dim],
                    )
            except Exception:
                roll_fake = None

        with graph.inserting_before(cat):
            roll_node = graph.call_function(
                torch.ops.aten.roll.default,
                args=(parent, [shift], [cat_dim]),
            )
            if roll_fake is not None:
                roll_node.meta["val"] = roll_fake
            elif "val" in cat.meta:
                roll_node.meta["val"] = cat.meta["val"]

        cat.replace_all_uses_with(roll_node)
        changed = True
        log.info(
            "cat_to_view_pass: collapsed cat(%d slices, dim=%d) of %s "
            "-> roll(shift=%d) (cyclic rotation of full cover [0, %d))",
            len(cat_inputs),
            cat_dim,
            parent.name,
            shift,
            dim_size,
        )

    eliminate_dead_code(graph, changed, cat_to_view_pass.__name__)


_REPEAT_BROADCAST_FRIENDLY_OPS = frozenset(
    (
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.where.self,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.logical_and.default,
        torch.ops.aten.logical_or.default,
        torch.ops.aten.logical_xor.default,
        torch.ops.aten.logical_not.default,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.bitwise_not.default,
        torch.ops.aten.maximum.default,
        torch.ops.aten.minimum.default,
        torch.ops.aten.fmod.Tensor,
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.masked_fill.Scalar,
        torch.ops.aten.masked_fill.Tensor,
    )
)


@register_custom_pass(PassType.POST)
def repeat_to_expand_pass(graph: torch.fx.Graph) -> None:
    """Rewrite a broadcast-only repeat as expand: when every user supports broadcasting,
    replace the physically copying repeat with a zero-copy expand."""
    target_repeat = torch.ops.aten.repeat.default
    changed = False

    for rpt in list(graph.nodes):
        if rpt.op != "call_function" or rpt.target is not target_repeat:
            continue
        if len(rpt.args) < 2:
            continue
        inp = rpt.args[0]
        repeats = rpt.args[1]
        if not isinstance(inp, torch.fx.Node):
            continue
        if not isinstance(repeats, (list, tuple)):
            continue

        in_shape = get_node_shape(inp, allow_symbolic=True)
        if in_shape is None:
            continue
        if len(repeats) != len(in_shape):
            continue

        # Only broadcast (no physical copy) can be rewritten: each dim either is
        # not repeated (r==1, output keeps the original dim, may be symbolic) or the
        # original dim is provably 1 (output dim is r). repeats must be int constants.
        valid = True
        out_shape = []
        for r, s in zip(repeats, in_shape):
            r = resolve_size_arg(r)
            if not isinstance(r, int):
                valid = False
                break
            if r == 1:
                out_shape.append(s)
            elif is_statically_one(s):
                out_shape.append(r)
            else:
                valid = False
                break
        if not valid:
            continue

        users_ok = all(
            (u.op == "call_function" and u.target in _REPEAT_BROADCAST_FRIENDLY_OPS)
            for u in rpt.users
        )
        if not users_ok or not list(rpt.users):
            continue

        inp_fake = inp.meta.get("val")
        fake_mode = (
            getattr(inp_fake, "fake_mode", None) if inp_fake is not None else None
        )

        with graph.inserting_before(rpt):
            # Symbolic dims in the output shape all come from inp itself; materialize as sym_size refs.
            expand_shape = materialize_shape(graph, out_shape, inp)
            if expand_shape is None:
                continue
            exp = graph.call_function(
                torch.ops.aten.expand.default,
                args=(inp, expand_shape),
            )
        if "val" in rpt.meta:
            exp.meta["val"] = rpt.meta["val"]
        elif fake_mode is not None and inp_fake is not None:
            try:
                with fake_mode:
                    exp.meta["val"] = torch.ops.aten.expand.default(
                        inp_fake,
                        out_shape,
                    )
            except Exception:
                pass

        rpt.replace_all_uses_with(exp)
        changed = True
        log.info(
            "repeat_to_expand_pass: rewrote repeat(%s, %s) -> "
            "expand(%s, %s) (broadcast-only, %d consumer%s)",
            inp.name,
            list(repeats),
            inp.name,
            out_shape,
            len(exp.users),
            "" if len(exp.users) == 1 else "s",
        )

    eliminate_dead_code(graph, changed, repeat_to_expand_pass.__name__)


_IOTA_DTYPE_TRANSPARENT_OPS = frozenset(
    (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.expand.default,
        torch.ops.aten.broadcast_to.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.abs.default,
    )
)

_IOTA_DTYPE_CLOSING_OPS = frozenset(
    (
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    )
)


def _prims_iota_endpoints(node):
    """Return the two endpoints (start, last) of a prims.iota sequence (may be symbolic);
    all elements lie within [min(start,last), max(start,last)]. None if args are unresolvable."""
    if not (
        node.op == "call_function"
        and node.target is torch.ops.prims.iota.default
        and node.args
    ):
        return None
    length = resolve_size_arg(node.args[0])
    start = resolve_size_arg(node.kwargs.get("start", 0))
    step = resolve_size_arg(node.kwargs.get("step", 1))
    if length is None or start is None or step is None:
        return None
    if isinstance(length, int) and length <= 0:
        return (start, start)
    last = start + (length - 1) * step
    return (start, last)


def _collect_iota_downcast_closure(iota_node):
    """Collect every intermediate node from iota through dtype-transparent ops up to a
    terminating op; return None if an unsupported op appears and narrowing is unsafe."""
    middle_ids = OrderedSet()
    queue = [iota_node]
    while queue:
        cur = queue.pop(0)
        for user in cur.users:
            if user.op != "call_function":
                return None
            t = user.target
            if t in _IOTA_DTYPE_CLOSING_OPS:
                continue
            if t in _IOTA_DTYPE_TRANSPARENT_OPS:
                if id(user) in middle_ids:
                    continue
                middle_ids.add(id(user))
                queue.append(user)
                continue
            return None
    return middle_ids


def _hashable_const_key(value):
    """Turn constant args (including nested list/tuple/dict) into a hashable key for CSE."""
    if isinstance(value, list):
        return ("__list__",) + tuple(_hashable_const_key(v) for v in value)
    if isinstance(value, tuple):
        return ("__tuple__",) + tuple(_hashable_const_key(v) for v in value)
    if isinstance(value, dict):
        return ("__dict__",) + tuple(
            sorted((k, _hashable_const_key(v)) for k, v in value.items())
        )
    hash(value)
    return value


def _collect_mutation_buffer_ids(graph):
    """Collect the ids of every node referenced by a mutation or triton kernel op.
    Matches triton_kernel_wrapper_mutation, triton_kernel_wrapper_functional and
    ATen in-place ops (detected through both __name__ and _schema.name).
    Walks args/kwargs (including nested dict/list) rather than relying on node.users."""

    def _is_mutation_node(n):
        if n.op != "call_function":
            return False
        target = n.target
        name = getattr(target, "__name__", "") or ""
        if name.startswith("triton_kernel_wrapper") or name.endswith("_"):
            return True
        if isinstance(target, torch._ops.OpOverload):
            schema_name = getattr(getattr(target, "_schema", None), "name", "")
            if schema_name.endswith("_"):
                return True
        return False

    def _collect_nodes(val, out):
        if isinstance(val, torch.fx.Node):
            out.add(id(val))
        elif isinstance(val, dict):
            for v in val.values():
                _collect_nodes(v, out)
        elif isinstance(val, (list, tuple)):
            for v in val:
                _collect_nodes(v, out)

    ids = set()
    for n in graph.nodes:
        if not _is_mutation_node(n):
            continue
        for a in n.args:
            _collect_nodes(a, ids)
        for v in n.kwargs.values():
            _collect_nodes(v, ids)
    return ids


def _cse_constant_call(graph, target, mutation_buf_ids=None):
    """Common subexpression elimination over constant-argument calls of a given target:
    repeated calls with the same args/kwargs keep the first and reuse its result.
    Nodes referenced by mutations are excluded, so several mutations never share a buffer."""
    if mutation_buf_ids is None:
        mutation_buf_ids = _collect_mutation_buffer_ids(graph)
    seen = {}
    changed = False
    for n in list(graph.nodes):
        if n.op != "call_function" or n.target is not target:
            continue
        if id(n) in mutation_buf_ids:
            continue
        try:
            key = (
                tuple(_hashable_const_key(a) for a in n.args),
                tuple(sorted((k, _hashable_const_key(v)) for k, v in n.kwargs.items())),
            )
        except TypeError:
            continue
        if key in seen:
            n.replace_all_uses_with(seen[key])
            changed = True
        else:
            seen[key] = n
    return changed


@register_custom_pass(PassType.POST)
def fold_iota_arithmetic_pass(graph: torch.fx.Graph) -> None:
    """Run constant CSE over iota/arange/full and try to narrow an int64 iota whose range
    fits in int32; also simplify cmp(sub(a,b),0) into cmp(a,b)."""
    changed = False

    iota_target = torch.ops.prims.iota.default
    mutation_buf_ids = _collect_mutation_buffer_ids(graph)
    for tgt in (
        iota_target,
        torch.ops.aten.arange.default,
        torch.ops.aten.full.default,
    ):
        if _cse_constant_call(graph, tgt, mutation_buf_ids):
            changed = True

    for iota in list(graph.nodes):
        if iota.op != "call_function" or iota.target is not iota_target:
            continue
        if iota.kwargs.get("dtype") is not torch.int64:
            continue
        endpoints = _prims_iota_endpoints(iota)
        if endpoints is None:
            continue
        if not statically_fits_int32(*endpoints):
            continue

        fake = iota.meta.get("val")
        fake_mode = getattr(fake, "fake_mode", None) if fake is not None else None
        if fake_mode is None:
            continue

        middle_ids = _collect_iota_downcast_closure(iota)
        if middle_ids is None:
            continue

        new_kwargs = dict(iota.kwargs)
        new_kwargs["dtype"] = torch.int32
        iota.kwargs = new_kwargs
        refresh_fake_meta(iota, fake_mode)
        if iota.meta.get("val") is None or iota.meta["val"].dtype is not torch.int32:
            new_kwargs["dtype"] = torch.int64
            iota.kwargs = new_kwargs
            refresh_fake_meta(iota, fake_mode)
            continue

        middle_nodes_in_topo = [n for n in graph.nodes if id(n) in middle_ids]
        for n in middle_nodes_in_topo:
            refresh_fake_meta(n, fake_mode)
        for n in list(graph.nodes):
            if (
                n.op == "call_function"
                and n.target in _IOTA_DTYPE_CLOSING_OPS
                and any(u is iota or id(u) in middle_ids for u in n.all_input_nodes)
            ):
                refresh_fake_meta(n, fake_mode)

        changed = True
        log.info(
            "fold_iota_arithmetic_pass: downcast iota int64 -> int32"
            " (%d transparent user%s in closure)",
            len(middle_ids),
            "" if len(middle_ids) == 1 else "s",
        )

    cmp_scalar_to_tensor = {
        torch.ops.aten.ge.Scalar: torch.ops.aten.ge.Tensor,
        torch.ops.aten.gt.Scalar: torch.ops.aten.gt.Tensor,
        torch.ops.aten.le.Scalar: torch.ops.aten.le.Tensor,
        torch.ops.aten.lt.Scalar: torch.ops.aten.lt.Tensor,
        torch.ops.aten.eq.Scalar: torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Scalar: torch.ops.aten.ne.Tensor,
    }

    for cmp in list(graph.nodes):
        if cmp.op != "call_function" or cmp.target not in cmp_scalar_to_tensor:
            continue
        if len(cmp.args) < 2:
            continue
        sub = cmp.args[0]
        rhs = cmp.args[1]
        if not (isinstance(rhs, (int, float)) and rhs == 0):
            continue
        if not (
            isinstance(sub, torch.fx.Node)
            and sub.op == "call_function"
            and sub.target == torch.ops.aten.sub.Tensor
        ):
            continue
        if len(sub.args) < 2:
            continue
        alpha = sub.kwargs.get("alpha", 1)
        if alpha != 1:
            continue

        a, b = sub.args[0], sub.args[1]
        # the identity cmp(a - b, 0) == cmp(a, b) holds whether b is a tensor or a scalar,
        # but the aten comparison overload must match: use .Tensor when b is a tensor and
        # keep .Scalar when b is a scalar (int/float), otherwise this raises
        # "Expected a value of type 'Tensor' ... but instead found type 'float'".
        b_is_tensor = isinstance(b, torch.fx.Node) and isinstance(
            b.meta.get("val"), torch.Tensor
        )
        if b_is_tensor:
            new_target = cmp_scalar_to_tensor[cmp.target]
        elif isinstance(b, (int, float, bool)):
            new_target = cmp.target
        else:
            continue
        with graph.inserting_before(cmp):
            new_cmp = graph.call_function(new_target, args=(a, b))
        if "val" in cmp.meta:
            new_cmp.meta["val"] = cmp.meta["val"]
        cmp.replace_all_uses_with(new_cmp)
        changed = True
        log.info(
            "fold_iota_arithmetic_pass: folded %s(sub(a, b), 0) -> %s(a, b)",
            cmp.target,
            new_target,
        )

    eliminate_dead_code(graph, changed, fold_iota_arithmetic_pass.__name__)


def _extract_const_full_scalar(node):
    """Extract the constant fill scalar from an aten.full node;
    return None if the node is not a full or the fill value is not a scalar."""
    if not (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target == torch.ops.aten.full.default
    ):
        return None
    if len(node.args) < 2:
        return None
    v = node.args[1]
    return v if isinstance(v, (int, float, bool)) else None


@register_custom_pass(PassType.POST)
def broadcast_const_mask_compress(graph: torch.fx.Graph) -> None:
    """Collapse cast(where(bool_mask, full(c1), full(c2))): when the two constants form a
    0/1 choice, replace the where+cast with the mask itself (or logical_not(mask))."""
    changed = False

    for node in list(graph.nodes):
        if not is_cast_node(node):
            continue
        target_dtype = normalize_dtype(get_cast_dtype(node))
        if target_dtype is None:
            continue

        w = node.args[0] if node.args else None
        if not (
            isinstance(w, torch.fx.Node)
            and w.op == "call_function"
            and w.target == torch.ops.aten.where.self
        ):
            continue
        if len(w.args) < 3:
            continue

        w_users = [u for u in w.users if u.op != "output"]
        if len(w_users) != 1 or w_users[0] is not node:
            continue

        cond, full_t, full_f = w.args[0], w.args[1], w.args[2]
        if not (isinstance(cond, torch.fx.Node) and get_node_dtype(cond) == torch.bool):
            continue

        t_val = _extract_const_full_scalar(full_t)
        f_val = _extract_const_full_scalar(full_f)
        if t_val is None or f_val is None:
            continue

        t_true = bool(t_val)
        f_true = bool(f_val)
        if t_true == f_true:
            continue

        if target_dtype != torch.bool:
            if not ((t_val == 1 and f_val == 0) or (t_val == 0 and f_val == 1)):
                continue

        if t_true:
            new_cond = cond
            replacement_kind = "mask"
        else:
            cond_fake = cond.meta.get("val")
            fake_mode = (
                getattr(cond_fake, "fake_mode", None) if cond_fake is not None else None
            )

            with graph.inserting_before(node):
                new_cond = graph.call_function(
                    torch.ops.aten.logical_not.default,
                    args=(cond,),
                )
            if fake_mode is not None:
                try:
                    with fake_mode:
                        new_cond.meta["val"] = torch.ops.aten.logical_not.default(
                            cond_fake
                        )
                except Exception:
                    pass
            replacement_kind = "logical_not(mask)"

        if target_dtype == torch.bool:
            node.replace_all_uses_with(new_cond)
            action = "drop cast, substitute mask"
        else:
            node.replace_input_with(w, new_cond)
            action = "rewire cast input from where -> mask"

        changed = True

        log.info(
            "broadcast_const_mask_compress: collapsed "
            "cast[%s](where(bool_mask, full(%s), full(%s))) -> %s "
            "(%s; dropping explicit broadcast to %s)",
            target_dtype,
            t_val,
            f_val,
            replacement_kind,
            action,
            get_node_shape(w, allow_symbolic=True),
        )

    eliminate_dead_code(graph, changed, broadcast_const_mask_compress.__name__)


def _is_zero_tensor_source(node):
    """Tell whether a node is an all-zero source: scalar 0, or zeros / zeros_like / full(0)."""
    if isinstance(node, (int, float)) and node == 0:
        return True
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op != "call_function":
        return False
    if node.target in (
        torch.ops.aten.zeros.default,
        torch.ops.aten.zeros_like.default,
    ):
        return True
    if node.target is torch.ops.aten.full.default:
        fill = node.args[1] if len(node.args) > 1 else None
        return isinstance(fill, (int, float, bool)) and fill == 0
    return False


def _strip_logical_not(node):
    """Strip the outermost logical / bitwise not and return (inner node, was_negated);
    if there is no not, return the node unchanged with False."""
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return node, False
    if node.target is torch.ops.aten.logical_not.default:
        return node.args[0], True
    if (
        node.target is torch.ops.aten.bitwise_not.default
        and get_node_dtype(node.args[0]) == torch.bool
    ):
        return node.args[0], True
    return node, False


def _are_logically_negated_masks(m1, m2):
    """Tell whether two masks are exact complements: same base node, not on one side only."""
    if m1 is m2:
        return False
    s1, neg1 = _strip_logical_not(m1)
    s2, neg2 = _strip_logical_not(m2)
    return s1 is s2 and (neg1 ^ neg2)


def _match_masked_zero_where(node):
    """Match where(mask, val, 0) and return (mask, val), otherwise None."""
    if not (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.where.self
        and len(node.args) == 3
    ):
        return None
    cond, val, other = node.args
    if not isinstance(cond, torch.fx.Node):
        return None
    if not isinstance(val, torch.fx.Node):
        return None
    if not _is_zero_tensor_source(other):
        return None
    return cond, val


@register_custom_pass(PassType.POST)
def masked_add_compose_pass(graph: torch.fx.Graph) -> None:
    """Fuse where(m, a, 0) + where(~m, b, 0) into a single where(m, a, b):
    adding two complementary masked values equals one select, saving an add and a where."""
    changed = False

    for add in list(graph.nodes):
        if add.op != "call_function" or add.target is not torch.ops.aten.add.Tensor:
            continue
        if len(add.args) < 2:
            continue
        alpha = add.kwargs.get("alpha", 1)
        if alpha != 1:
            continue

        lhs_match = _match_masked_zero_where(add.args[0])
        rhs_match = _match_masked_zero_where(add.args[1])
        if lhs_match is None or rhs_match is None:
            continue
        w_lhs, w_rhs = add.args[0], add.args[1]
        if len(w_lhs.users) != 1 or len(w_rhs.users) != 1:
            continue
        m_lhs, v_lhs = lhs_match
        m_rhs, v_rhs = rhs_match
        if not _are_logically_negated_masks(m_lhs, m_rhs):
            continue

        _, m_lhs_is_neg = _strip_logical_not(m_lhs)
        if m_lhs_is_neg:
            mask_pos, val_pos, val_neg = m_rhs, v_rhs, v_lhs
        else:
            mask_pos, val_pos, val_neg = m_lhs, v_lhs, v_rhs

        pos_fake = val_pos.meta.get("val")
        fake_mode = (
            getattr(pos_fake, "fake_mode", None) if pos_fake is not None else None
        )

        with graph.inserting_before(add):
            new_where = graph.call_function(
                torch.ops.aten.where.self,
                args=(mask_pos, val_pos, val_neg),
            )
        if fake_mode is not None:
            try:
                mp_fake = mask_pos.meta.get("val")
                vp_fake = val_pos.meta.get("val")
                vn_fake = val_neg.meta.get("val")
                if None not in (mp_fake, vp_fake, vn_fake):
                    with fake_mode:
                        new_where.meta["val"] = torch.ops.aten.where.self(
                            mp_fake, vp_fake, vn_fake
                        )
            except Exception:
                pass
        if "val" not in new_where.meta and "val" in add.meta:
            new_where.meta["val"] = add.meta["val"]

        add.replace_all_uses_with(new_where)
        changed = True
        log.info(
            "masked_add_compose_pass: folded "
            "where(m, a, 0) + where(~m, b, 0) -> where(m, a, b) "
            "(mask=%s)",
            mask_pos.name,
        )

    eliminate_dead_code(graph, changed, masked_add_compose_pass.__name__)


_BCM_VIEW_CHAIN_OPS = frozenset(
    (
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.broadcast_to.default,
        torch.ops.aten.flatten.using_ints,
    )
)


def _walk_back_view_chain_to_cast(node):
    """Walk backwards from node along a single-user view chain until a cast node:
    return (view chain, cast node), or (None, None) when the match fails."""
    chain = []
    cur = node
    visited = OrderedSet()
    while True:
        if not isinstance(cur, torch.fx.Node) or cur.op != "call_function":
            return None, None
        if id(cur) in visited:
            return None, None
        visited.add(id(cur))
        if is_cast_node(cur):
            return chain, cur
        if cur.target not in _BCM_VIEW_CHAIN_OPS:
            return None, None
        if len(cur.users) != 1:
            return None, None
        chain.append(cur)
        if not cur.args:
            return None, None
        cur = cur.args[0]


def _replay_view_chain(graph, fake_mode, base_node, chain):
    """Replay the given view chain in the same order starting from base_node,
    creating an equivalent node chain in the graph and refreshing its fake meta."""
    cur = base_node
    for view in reversed(chain):
        new_args = (cur,) + tuple(view.args[1:])
        new_node = graph.call_function(
            view.target,
            args=new_args,
            kwargs=dict(view.kwargs),
        )
        if fake_mode is not None:
            try:
                cur_fake = cur.meta.get("val")
                if cur_fake is not None:

                    def _resolve(arg, fm=fake_mode):
                        if isinstance(arg, torch.fx.Node):
                            return arg.meta.get("val", arg)
                        if isinstance(arg, (list, tuple)):
                            return type(arg)(_resolve(x, fm) for x in arg)
                        return arg

                    with fake_mode:
                        new_node.meta["val"] = view.target(
                            *[_resolve(a) for a in new_args],
                            **{k: _resolve(v) for k, v in view.kwargs.items()},
                        )
            except Exception:
                pass
        cur = new_node
    return cur


@register_custom_pass(PassType.POST)
def bool_cast_mul_to_where_pass(graph: torch.fx.Graph) -> None:
    """Rewrite cast[dtype](bool_mask) * x into where(bool_mask, x, 0):
    this drops an explicit bool->numeric cast and a broadcast multiply, leaving the backend more room to fuse."""
    changed = False

    for mul in list(graph.nodes):
        if mul.op != "call_function" or mul.target is not torch.ops.aten.mul.Tensor:
            continue
        if len(mul.args) != 2:
            continue
        a, b = mul.args

        a_is_node = isinstance(a, torch.fx.Node)
        b_is_node = isinstance(b, torch.fx.Node)

        chain, cast_node, other = None, None, None
        if a_is_node:
            chain_a, cast_a = _walk_back_view_chain_to_cast(a)
            if cast_a is not None and b_is_node:
                chain, cast_node, other = chain_a, cast_a, b
        if cast_node is None and b_is_node:
            chain_b, cast_b = _walk_back_view_chain_to_cast(b)
            if cast_b is not None and a_is_node:
                chain, cast_node, other = chain_b, cast_b, a
        if cast_node is None or other is None:
            continue

        if len(cast_node.users) != 1:
            continue

        cast_src = cast_node.args[0] if cast_node.args else None
        if not (
            isinstance(cast_src, torch.fx.Node)
            and get_node_dtype(cast_src) == torch.bool
        ):
            continue

        cast_target_dtype = normalize_dtype(get_cast_dtype(cast_node))
        other_dtype = get_node_dtype(other)
        if cast_target_dtype is None or other_dtype is None:
            continue
        if cast_target_dtype != other_dtype:
            continue

        other_fake = other.meta.get("val")
        if other_fake is None:
            continue
        fake_mode = getattr(other_fake, "fake_mode", None)
        if fake_mode is None:
            continue
        device = getattr(other_fake, "device", None)

        try:
            with fake_mode:
                zero_fake = torch.ops.aten.full.default(
                    [],
                    0,
                    dtype=other_dtype,
                    device=device,
                )
        except Exception:
            continue

        with graph.inserting_before(mul):
            new_cond = _replay_view_chain(
                graph,
                fake_mode,
                cast_src,
                chain,
            )
            zero_node = graph.call_function(
                torch.ops.aten.full.default,
                args=([], 0),
                kwargs={"dtype": other_dtype, "device": device},
            )
            zero_node.meta["val"] = zero_fake
            new_where = graph.call_function(
                torch.ops.aten.where.self,
                args=(new_cond, other, zero_node),
            )

        try:
            with fake_mode:
                nc_fake = new_cond.meta.get("val")
                if nc_fake is not None:
                    new_where.meta["val"] = torch.ops.aten.where.self(
                        nc_fake,
                        other_fake,
                        zero_fake,
                    )
        except Exception:
            pass
        if "val" not in new_where.meta and "val" in mul.meta:
            new_where.meta["val"] = mul.meta["val"]

        mul.replace_all_uses_with(new_where)
        changed = True
        chain_desc = " -> ".join(v.target.__name__ for v in chain) if chain else "direct"
        log.info(
            "bool_cast_mul_to_where_pass: folded "
            "cast[%s](bool_mask) * x -> where(bool_mask, x, 0) "
            "(mask=%s, view-chain=[%s], shape=%s)",
            cast_target_dtype,
            cast_src.name,
            chain_desc,
            get_node_shape(other, allow_symbolic=True),
        )

    eliminate_dead_code(graph, changed, bool_cast_mul_to_where_pass.__name__)


def _peel_single_user_relu_sign(node):
    """Match and strip a single-user relu(sign(x)) chain, returning the inner x, else None."""
    if not (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.relu.default
        and len(node.users) == 1
        and node.args
    ):
        return None
    sign = node.args[0]
    if not (
        isinstance(sign, torch.fx.Node)
        and sign.op == "call_function"
        and sign.target is torch.ops.aten.sign.default
        and len(sign.users) == 1
        and sign.args
    ):
        return None
    return sign.args[0]


@register_custom_pass(PassType.POST)
def sign_diff_hamming_fuse_pass(graph: torch.fx.Graph) -> None:
    """Fuse sum(abs(relu(sign(x)) - relu(sign(y)))) into a Hamming distance over sign bits:
    the equivalent sum(ne(gt(x,0), gt(y,0))) shortens the op chain and drops intermediates."""
    changed = False
    for sum_node in list(graph.nodes):
        if (
            sum_node.op != "call_function"
            or sum_node.target is not torch.ops.aten.sum.dim_IntList
        ):
            continue
        if len(sum_node.args) < 2:
            continue
        abs_node = sum_node.args[0]
        if not (
            isinstance(abs_node, torch.fx.Node)
            and abs_node.op == "call_function"
            and abs_node.target is torch.ops.aten.abs.default
            and len(abs_node.users) == 1
            and abs_node.args
        ):
            continue
        sub_node = abs_node.args[0]
        if not (
            isinstance(sub_node, torch.fx.Node)
            and sub_node.op == "call_function"
            and sub_node.target is torch.ops.aten.sub.Tensor
            and len(sub_node.users) == 1
            and len(sub_node.args) == 2
            and sub_node.kwargs.get("alpha", 1) == 1
        ):
            continue

        a, b = sub_node.args
        x_src = _peel_single_user_relu_sign(a)
        y_src = _peel_single_user_relu_sign(b)
        if x_src is None or y_src is None:
            continue
        # gt.Scalar requires self to be a tensor; the input of relu(sign(x)) always is one,
        # but guard explicitly so a scalar can never reach gt.Scalar(self, 0).
        if not (
            isinstance(x_src, torch.fx.Node) and isinstance(y_src, torch.fx.Node)
        ):
            continue

        out_dtype = get_node_dtype(sum_node)
        if out_dtype is None:
            continue

        dim_arg = sum_node.args[1]
        keepdim_arg = (
            sum_node.args[2]
            if len(sum_node.args) > 2
            else sum_node.kwargs.get("keepdim", False)
        )

        # Resolve fake_mode for meta refresh from any incoming tensor.
        src_fake = x_src.meta.get("val") if isinstance(x_src, torch.fx.Node) else None
        fake_mode = (
            getattr(src_fake, "fake_mode", None) if src_fake is not None else None
        )

        with graph.inserting_before(sum_node):
            gt_x = graph.call_function(torch.ops.aten.gt.Scalar, args=(x_src, 0))
            gt_y = graph.call_function(torch.ops.aten.gt.Scalar, args=(y_src, 0))
            ne_node = graph.call_function(torch.ops.aten.ne.Tensor, args=(gt_x, gt_y))
            new_sum = graph.call_function(
                torch.ops.aten.sum.dim_IntList,
                args=(ne_node, dim_arg, keepdim_arg),
                kwargs={"dtype": out_dtype},
            )

        if fake_mode is not None:
            for n in (gt_x, gt_y, ne_node, new_sum):
                refresh_fake_meta(n, fake_mode)
        if "val" not in new_sum.meta and "val" in sum_node.meta:
            new_sum.meta["val"] = sum_node.meta["val"]

        sum_node.replace_all_uses_with(new_sum)
        changed = True
        log.info(
            "sign_diff_hamming_fuse_pass: folded "
            "sum(abs(sub(relu(sign(%s)), relu(sign(%s))))) -> "
            "sum(ne(gt(.,0), gt(.,0)), dtype=%s) "
            "(dim=%s, keepdim=%s)",
            x_src.name if isinstance(x_src, torch.fx.Node) else x_src,
            y_src.name if isinstance(y_src, torch.fx.Node) else y_src,
            out_dtype,
            dim_arg,
            keepdim_arg,
        )

    eliminate_dead_code(graph, changed, sign_diff_hamming_fuse_pass.__name__)


def _has_default_embedding_args(node):
    """Tell whether an aten.embedding call uses all default arguments
    (padding_idx=-1, scale_grad_by_freq and sparse both off)."""
    if len(node.args) > 2 and node.args[2] != -1:
        return False
    if len(node.args) > 3 and node.args[3]:
        return False
    if len(node.args) > 4 and node.args[4]:
        return False
    for k, v in node.kwargs.items():
        if k == "padding_idx" and v != -1:
            return False
        if k in ("scale_grad_by_freq", "sparse") and v:
            return False
    return True


def _symbolic_shape_key(shape):
    """Build a hashable shape key: SymInt dims use the canonical sympy expr string
    (so s0+s0 and 2*s0 group together), other dims become int, for embedding grouping."""

    def _dim_key(d):
        if isinstance(d, torch.SymInt):
            try:
                return f"sym:{d.node.expr}"
            except Exception:
                return f"sym:{d}"
        return int(d)

    return tuple(_dim_key(d) for d in shape)


def _weight_node_key(w):
    """Build an identity key for an embedding weight: params/constants share (op, target), others go by id."""
    if w.op in ("get_attr", "placeholder"):
        return (w.op, w.target)
    return id(w)


_REDUCE_OPS_DIM_LIST = {
    torch.ops.aten.sum.dim_IntList: "sum",
    torch.ops.aten.mean.dim: "mean",
    torch.ops.aten.amax.default: "amax",
    torch.ops.aten.amin.default: "amin",
}

_REDUCE_OPS_DIM_INT = {
    torch.ops.aten.prod.dim_int: "prod",
}


def _reduce_call_args(target, input_node, reduce_dim):
    """Build the reduce dim argument from the op signature: dim_int takes an int, dim_list a list."""
    if target in _REDUCE_OPS_DIM_INT:
        return (input_node, reduce_dim)
    return (input_node, [reduce_dim])


def _detect_reduce_pattern(nodes, cat_dim):
    """Check that every embedding node's sole user is the same reduce along cat_dim with keepdim=False;
    return (reduce_target, reduce_nodes) when they agree, otherwise None."""
    reduce_target = None
    reduce_nodes = []
    for node in nodes:
        users = [u for u in node.users if u.op != "output"]
        if len(users) != 1:
            return None
        user = users[0]
        if user.op != "call_function":
            return None

        target = user.target
        if target in _REDUCE_OPS_DIM_LIST:
            dims = user.args[1] if len(user.args) > 1 else None
            if dims != [cat_dim]:
                return None
            keepdim = (
                user.args[2]
                if len(user.args) > 2
                else user.kwargs.get("keepdim", False)
            )
        elif target in _REDUCE_OPS_DIM_INT:
            dim_arg = user.args[1] if len(user.args) > 1 else None
            if dim_arg != cat_dim:
                return None
            keepdim = (
                user.args[2]
                if len(user.args) > 2
                else user.kwargs.get("keepdim", False)
            )
        else:
            return None

        if keepdim:
            return None
        if reduce_target is None:
            reduce_target = target
        elif reduce_target is not target:
            return None
        reduce_nodes.append(user)
    return reduce_target, reduce_nodes


def _detect_indices_parent(nodes):
    """Detect whether the indices of a group of embedding nodes are contiguous slices of one
    parent along the same dim, covering it fully without overlap; if so return (parent, slice_dim)."""
    if not nodes:
        return None, None

    parent = None
    slice_dim = None
    slices_info = []

    for node in nodes:
        idx = node.args[1]
        if not (
            isinstance(idx, torch.fx.Node)
            and idx.op == "call_function"
            and idx.target == torch.ops.aten.slice.Tensor
        ):
            return None, None

        src = idx.args[0]
        dim = idx.args[1] if len(idx.args) > 1 else 0
        start = idx.args[2] if len(idx.args) > 2 else 0
        end = idx.args[3] if len(idx.args) > 3 else None

        if isinstance(dim, torch.fx.Node):
            return None, None
        dim = int(dim)

        if parent is None:
            parent = src
            slice_dim = dim
        elif src is not parent or dim != slice_dim:
            return None, None

        if isinstance(start, torch.fx.Node) or isinstance(end, torch.fx.Node):
            return None, None
        slices_info.append((int(start) if start is not None else 0, end))

    parent_shape = get_node_shape(parent, allow_symbolic=True)
    if parent_shape is None or slice_dim >= len(parent_shape):
        return None, None
    dim_size = parent_shape[slice_dim]
    # The sliced field dim must be static (the N*L coverage proof needs a concrete length); batch dim may be symbolic.
    if isinstance(dim_size, torch.SymInt):
        return None, None
    dim_size = int(dim_size)

    resolved = [(s, int(e) if e is not None else dim_size) for s, e in slices_info]
    resolved_sorted = sorted(resolved, key=lambda x: x[0])
    expected = 0
    for s, e in resolved_sorted:
        if s != expected:
            return None, None
        expected = e
    if expected != dim_size:
        return None, None

    return parent, slice_dim


def _fuse_embedding_subgroup(graph, nodes, node_order, D):
    """Batch-fuse embedding calls sharing a weight: after checking the index source and the
    downstream reduce pattern, forward to the Pattern C reshape-first implementation."""
    if len(nodes) < 2:
        return False

    weight_node = nodes[0].args[0]
    w_fake = weight_node.meta.get("val")
    if w_fake is None:
        return False
    V = int(get_node_shape(weight_node)[0])

    fake_mode = getattr(w_fake, "fake_mode", None)
    if fake_mode is None:
        return False

    parent_node, slice_dim = _detect_indices_parent(nodes)
    if parent_node is None:
        return False
    parent_fake = parent_node.meta.get("val")
    if parent_fake is None:
        return False

    reduce_info = _detect_reduce_pattern(nodes, slice_dim)
    if reduce_info is None:
        return False
    reduce_target, reduce_nodes = reduce_info

    return _apply_pattern_c_reshape_first(
        graph,
        nodes,
        reduce_nodes,
        weight_node,
        parent_node,
        reduce_target,
        slice_dim,
        fake_mode,
        w_fake,
        parent_fake,
        V,
        D,
    )


def _try_collapse_select_chain_into_cat_reshape(
    graph,
    source_node,
    source_fake,
    ordered_chain_nodes,
    source_axis,
    fake_mode,
):
    """Try to collapse a downstream chain of N selects feeding one cat into a reshape:
    when the selects appear in order along the cat's last dim, a flattening reshape feeds the cat directly."""
    src_shape = get_node_shape(source_node)
    if src_shape is None:
        return False
    ndim = len(src_shape)
    if source_axis != ndim - 2:
        return False

    cat_node = None
    for cn in ordered_chain_nodes:
        if len(cn.users) != 1:
            return False
        user = next(iter(cn.users.keys()))
        if user.op != "call_function" or user.target != torch.ops.aten.cat.default:
            return False
        if cat_node is None:
            cat_node = user
        elif user is not cat_node:
            return False
    if cat_node is None:
        return False

    cat_args_in = cat_node.args
    cat_inputs = list(cat_args_in[0])
    cat_dim = cat_args_in[1] if len(cat_args_in) > 1 else 0
    chain_out_ndim = ndim - 1
    if cat_dim not in (-1, chain_out_ndim - 1):
        return False

    for cn in ordered_chain_nodes:
        if sum(1 for x in cat_inputs if x is cn) != 1:
            return False

    N = len(ordered_chain_nodes)
    try:
        start = cat_inputs.index(ordered_chain_nodes[0])
    except ValueError:
        return False
    if start + N > len(cat_inputs):
        return False
    for i in range(N):
        if cat_inputs[start + i] is not ordered_chain_nodes[i]:
            return False

    D = src_shape[-1]
    new_shape = list(src_shape[:-2]) + [src_shape[-2] * D]
    try:
        with fake_mode:
            flat_fake = torch.ops.aten.reshape.default(
                source_fake,
                list(new_shape),
            )
    except Exception:
        return False

    with graph.inserting_after(source_node):
        flat = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(source_node, list(new_shape)),
        )
        flat.meta["val"] = flat_fake

    new_cat_inputs = cat_inputs[:start] + [flat] + cat_inputs[start + N :]
    cat_node.args = (new_cat_inputs,) + tuple(cat_args_in[1:])
    return True


def _apply_pattern_c_reshape_first(
    graph,
    nodes,
    reduce_nodes,
    weight_node,
    parent_node,
    reduce_target,
    slice_dim,
    fake_mode,
    w_fake,
    parent_fake,
    V,
    D,
):
    """Pattern C batch embedding fusion: reshape the parent indices to add an extra N dim,
    run one embedding + reduce, then reconnect the original users with select (or a collapsed cat)."""
    N = len(nodes)

    parent_shape = get_node_shape(parent_node, allow_symbolic=True)
    if parent_shape is None:
        return False
    if slice_dim >= len(parent_shape):
        return False
    dim_size_raw = parent_shape[slice_dim]
    if isinstance(dim_size_raw, torch.SymInt):
        return False
    dim_size = int(dim_size_raw)

    lengths = OrderedSet()
    starts = []
    for node in nodes:
        idx = node.args[1]
        start = idx.args[2] if len(idx.args) > 2 else 0
        end = idx.args[3] if len(idx.args) > 3 else None
        if isinstance(start, torch.fx.Node) or isinstance(end, torch.fx.Node):
            return False
        start_i = int(start) if start is not None else 0
        end_i = int(end) if end is not None else dim_size
        lengths.add(end_i - start_i)
        starts.append((node, start_i))
    if len(lengths) != 1:
        return False
    L = lengths.pop()
    if L <= 0 or N * L != dim_size:
        return False

    new_idx_shape = list(parent_shape)
    new_idx_shape[slice_dim] = N
    new_idx_shape.insert(slice_dim + 1, L)
    new_reduce_dim = slice_dim + 1

    order = sorted(range(N), key=lambda i: starts[i][1])
    ordered_emb_nodes = [nodes[i] for i in order]
    ordered_reduce_nodes = [reduce_nodes[i] for i in order]

    try:
        with fake_mode:
            new_idx_fake = torch.ops.aten.reshape.default(
                parent_fake,
                list(new_idx_shape),
            )
            new_emb_fake = torch.ops.aten.embedding.default(
                w_fake,
                new_idx_fake,
            )
            new_reduce_fake = reduce_target(
                *_reduce_call_args(reduce_target, new_emb_fake, new_reduce_dim),
            )
    except Exception:
        return False

    with graph.inserting_before(ordered_emb_nodes[0]):
        # Symbolic dims of new_idx_shape (e.g. batch dim) all come from parent;
        # materialize as sym_size refs to parent instead of writing raw SymInt into node args.
        materialized_idx_shape = materialize_shape(
            graph, list(new_idx_shape), parent_node
        )
        if materialized_idx_shape is None:
            return False
        reshaped_parent = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(parent_node, materialized_idx_shape),
        )
        reshaped_parent.meta["val"] = new_idx_fake

        new_emb = graph.call_function(
            torch.ops.aten.embedding.default,
            args=(weight_node, reshaped_parent),
        )
        new_emb.meta["val"] = new_emb_fake

        new_reduce = graph.call_function(
            reduce_target,
            args=_reduce_call_args(reduce_target, new_emb, new_reduce_dim),
        )
        new_reduce.meta["val"] = new_reduce_fake

    collapsed = _try_collapse_select_chain_into_cat_reshape(
        graph,
        new_reduce,
        new_reduce_fake,
        ordered_reduce_nodes,
        slice_dim,
        fake_mode,
    )

    if not collapsed:
        for i, rn in enumerate(ordered_reduce_nodes):
            try:
                with fake_mode:
                    sel_fake = torch.ops.aten.select.int(
                        new_reduce_fake,
                        slice_dim,
                        i,
                    )
            except Exception:
                return False

            with graph.inserting_before(rn):
                sel = graph.call_function(
                    torch.ops.aten.select.int,
                    args=(new_reduce, slice_dim, i),
                )
                sel.meta["val"] = sel_fake

            rn.replace_all_uses_with(sel)

    for rn in reversed(ordered_reduce_nodes):
        if len(rn.users) == 0:
            graph.erase_node(rn)
    for en in reversed(ordered_emb_nodes):
        if len(en.users) == 0:
            graph.erase_node(en)

    if collapsed:
        log.info(
            "batch_embedding_fusion_pass: Pattern C "
            "(reshape-first + cat-collapse) - replaced %d "
            "(slice+emb+reduce) + %d downstream selects with "
            "1 reshape + 1 emb + 1 reduce + 1 reshape "
            "(N=%d, L=%d, V=%d, D=%d)",
            N,
            N,
            N,
            L,
            V,
            D,
        )
    else:
        log.info(
            "batch_embedding_fusion_pass: Pattern C (reshape-first) "
            "- replaced %d (slice+emb+reduce) with "
            "1 reshape + 1 emb + 1 reduce + %d select "
            "(N=%d, L=%d, V=%d, D=%d)",
            N,
            N,
            N,
            L,
            V,
            D,
        )
    return True


@register_custom_pass(PassType.POST)
def batch_embedding_fusion_pass(graph: torch.fx.Graph) -> None:
    """Batch-fuse embedding calls: group by weight and index shape, then merge each group's
    repeated embedding+reduce into one reshape->embedding->reduce, cutting dispatch and memory cost."""
    changed = False

    emb_nodes = [
        n
        for n in graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.embedding.default
    ]
    if len(emb_nodes) < 2:
        return

    groups = {}
    for node in emb_nodes:
        if not _has_default_embedding_args(node):
            continue

        weight = node.args[0]
        indices = node.args[1]

        # The weight table (V, D) must be static; indices may contain symbolic dims (e.g. dynamic batch).
        w_shape = get_node_shape(weight)
        idx_shape = get_node_shape(indices, allow_symbolic=True)
        if w_shape is None or idx_shape is None or len(w_shape) != 2:
            continue
        if len(idx_shape) < 1:
            continue
        if any(isinstance(d, torch.SymInt) for d in w_shape):
            continue

        D = int(w_shape[1])
        key = (_weight_node_key(weight), D, _symbolic_shape_key(idx_shape))
        groups.setdefault(key, []).append(node)

    node_order = {n: i for i, n in enumerate(graph.nodes)}

    for key, all_nodes in groups.items():
        if len(all_nodes) < 2:
            continue
        all_nodes.sort(key=lambda n: node_order.get(n, 0))

        if _fuse_embedding_subgroup(graph, all_nodes, node_order, key[1]):
            changed = True

    eliminate_dead_code(graph, changed, batch_embedding_fusion_pass.__name__)


# the post_grad graph is already AOT-normalized to aten form: linear is decomposed into
# permute+addmm, so only addmm / mm / mm+add match here; w is already (K, N), no transpose.
_FMM_ADDMM_TARGETS = (torch.ops.aten.addmm.default,)
_FMM_MATMUL_TARGETS = (torch.ops.aten.mm.default,)
_FMM_ADD_TARGETS = (torch.ops.aten.add.Tensor,)
_FMM_RELU_TARGETS = (torch.ops.aten.relu.default,)

# fp32 would need HF32 on at call time, which the pass cannot guarantee, so only fp16/bf16.
_FMM_RELU_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_FMM_RELU_OP_TYPE = "relu"
_FMM_REQUIRED_RANK = 2


def _is_relu_node(node):
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in _FMM_RELU_TARGETS
    )


@functools.lru_cache(maxsize=1)
def _resolve_npu_fused_matmul():
    """Resolve the npu_fused_matmul OpOverload; return None so the pass degrades to no fusion.

    The post_grad graph must hold an OpOverload (torch.ops.npu.*) rather than the Python
    wrapper under torch_npu, otherwise inductor cannot route it as an extern fallback call.
    """
    if not is_ascend950:
        return None
    op = getattr(torch.ops.npu, "npu_fused_matmul", None)
    op = getattr(op, "default", None)
    if op is None:
        log.warning(
            "fused_matmul_relu_pass disabled: "
            "torch.ops.npu.npu_fused_matmul unavailable"
        )
    return op


def _fmm_node_meta(node):
    """Read a node's tensor meta. post_grad uses val, pre_grad (dynamo) uses example_value."""
    if not isinstance(node, torch.fx.Node):
        return None
    for key in ("val", "example_value", "tensor_meta"):
        meta = node.meta.get(key, None)
        if meta is not None and hasattr(meta, "dtype") and hasattr(meta, "shape"):
            return meta
    return None


def _is_provably_contiguous(meta):
    """Return False when undecidable; the npu_fused_matmul bias rejects non-contiguous input."""
    if meta is None or not hasattr(meta, "is_contiguous"):
        return False
    try:
        return bool(meta.is_contiguous())
    except Exception:
        return False


def _fused_matmul_form(node):
    """Identify which fusable form the node is: addmm / mm / add(mm), else None."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    if node.target in _FMM_ADDMM_TARGETS:
        return "addmm"
    if node.target in _FMM_MATMUL_TARGETS:
        return "mm"
    if node.target in _FMM_ADD_TARGETS:
        return "add"
    return None


def _is_matmul_node(node):
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in _FMM_MATMUL_TARGETS
    )


def _fmm_reject(node, reason):
    """Record why a match failed. Use INDUCTOR_ASCEND_LOG_LEVEL=DEBUG to see it."""
    log.debug("fused_matmul_relu_pass: skip %s, %s", node, reason)
    return None


def _fmm_is_mutation_node(node):
    """Whether the node writes in place. The name test matches _collect_mutation_buffer_ids,
    plus the schema write alias catches mutating ops that do not end in an underscore."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return False
    target = node.target
    name = getattr(target, "__name__", "") or ""
    if name.startswith("triton_kernel_wrapper") or name.endswith("_"):
        return True
    if not isinstance(target, torch._ops.OpOverload):
        return False
    try:
        schema = target._schema
        if schema.name.endswith("_"):
            return True
        return any(
            arg.alias_info is not None and arg.alias_info.is_write
            for arg in schema.arguments
        )
    except Exception:
        # undecidable when the schema is unreadable, so assume the worst and treat it as a mutation.
        return True


def _has_mutation_between(start, end):
    """Whether an in-place write happens strictly between start and end.

    The fused node is inserted at end but reads start's inputs; if anything overwrites them
    in that window the fusion reads the new values. If end does not follow start, return True.
    """
    node = start.next
    while node is not end:
        # the node list is circular, so reaching the sentinel root means end is not after start;
        # node is None only if the list is corrupted. Neither case can judge the window, so assume the worst.
        if node is None or node.op == "root":
            return True
        if _fmm_is_mutation_node(node):
            return True
        node = node.next
    return False


def _match_fused_matmul_relu_operands(node, form):
    """Match a fusable node and return (x1, x2, bias); bias is None when there is none.

    addmm(b, x, w) and mm(x, w) + b are both b + x @ w, while mm(x, w) has no bias;
    w is already (K, N), matching the op's x1 @ x2 + bias directly, so no transpose is needed.
    """
    if form == "addmm":
        if len(node.args) != 3:
            return _fmm_reject(node, "addmm operands passed as kwargs")
        # only fuse default beta/alpha, otherwise this is not equivalent to relu(x1@x2+bias).
        if node.kwargs.get("beta", 1) != 1 or node.kwargs.get("alpha", 1) != 1:
            return _fmm_reject(node, "addmm beta/alpha != 1")
        bias, x1, x2 = node.args
    elif form == "mm":
        if len(node.args) != 2:
            return _fmm_reject(node, "mm operands passed as kwargs")
        x1, x2 = node.args
        # the op's bias is optional, so leave it at the default None when absent.
        bias = None
    else:
        # add is commutative so the matmul may be on either side; an add with no matmul on either
        # side is unrelated to this pass, so skip it silently (relu(a + b) is common).
        if len(node.args) != 2:
            return None
        lhs, rhs = node.args
        if _is_matmul_node(lhs):
            mm_node, bias = lhs, rhs
        elif _is_matmul_node(rhs):
            mm_node, bias = rhs, lhs
        else:
            return None
        if node.kwargs.get("alpha", 1) != 1:
            return _fmm_reject(node, "add alpha != 1")
        if len(mm_node.users) != 1:
            return _fmm_reject(
                mm_node, f"has {len(mm_node.users)} users, expect only add"
            )
        if len(mm_node.args) != 2:
            return _fmm_reject(mm_node, "matmul operands passed as kwargs")
        # the fused node is inserted at the add, so an in-place write in the window would feed it rewritten x1/x2.
        if _has_mutation_between(mm_node, node):
            return _fmm_reject(node, "in-place write between mm and add")
        x1, x2 = mm_node.args

    if not (isinstance(x1, torch.fx.Node) and isinstance(x2, torch.fx.Node)):
        return _fmm_reject(node, "x1/x2 are not tensor nodes")
    x1_meta = _fmm_node_meta(x1)
    x2_meta = _fmm_node_meta(x2)
    if x1_meta is None or x2_meta is None:
        return _fmm_reject(node, "missing tensor meta on x1/x2")

    if x1_meta.dtype not in _FMM_RELU_SUPPORTED_DTYPES:
        return _fmm_reject(node, f"x1 dtype {x1_meta.dtype} not in fp16/bf16")
    if x2_meta.dtype != x1_meta.dtype:
        return _fmm_reject(
            node, f"dtype mismatch x1={x1_meta.dtype} x2={x2_meta.dtype}"
        )
    if len(x1_meta.shape) != _FMM_REQUIRED_RANK:
        return _fmm_reject(node, f"x1 shape {tuple(x1_meta.shape)} is not 2-D")
    if len(x2_meta.shape) != _FMM_REQUIRED_RANK:
        return _fmm_reject(node, f"weight shape {tuple(x2_meta.shape)} is not 2-D")

    if bias is None:
        return x1, x2, None
    if not isinstance(bias, torch.fx.Node):
        return _fmm_reject(node, "bias is not a tensor node")
    bias_meta = _fmm_node_meta(bias)
    if bias_meta is None:
        return _fmm_reject(node, "missing tensor meta on bias")
    if bias_meta.dtype != x1_meta.dtype:
        return _fmm_reject(
            node, f"dtype mismatch x1={x1_meta.dtype} bias={bias_meta.dtype}"
        )
    if len(bias_meta.shape) != 1:
        return _fmm_reject(node, f"bias shape {tuple(bias_meta.shape)} is not 1-D")
    if not _is_provably_contiguous(bias_meta):
        return _fmm_reject(node, "bias is not provably contiguous")
    return x1, x2, bias


def _fmm_peel_view(relu_node):
    """Strip one view between relu and matmul and return (view_node, mm_node).

    AOT splits a 3-D linear into view->addmm->view->relu->view, so relu sits on the view
    rather than the addmm. relu is pointwise and relu(view(t)) == view(relu(t)), so it can
    be absorbed into the fused op, letting the original view carry the output. Returns (None, relu's input) with no view.
    """
    inner = relu_node.args[0]
    if not check_view(inner):
        return None, inner
    if len(inner.users) != 1 or not inner.args:
        return None, inner
    return inner, inner.args[0]


@register_custom_pass(PassType.POST, ignore_inference_check=True)
def fused_matmul_relu_pass(graph: torch.fx.Graph) -> None:
    """Fuse relu(addmm) into npu_fused_matmul, which maps to aclnnFusedMatmul with
    fusedOpType="relu" (y = relu(x1@x2 + bias)).

    Pattern1: relu(addmm(bias, x, w))       -> npu_fused_matmul(x, w, bias, "relu")
    Pattern2: view(relu(view(addmm(...))))  -> view(npu_fused_matmul(...))
              AOT splits a 3-D linear into this shape with relu on the view; relu is
              pointwise, so absorb it into the op and let the original view carry the output.
    Pattern3: relu(mm(x, w) + bias)         -> npu_fused_matmul(x, w, bias, "relu")
              the shape seen when inductor did not fold mm+add into addmm.
    Pattern4: relu(mm(x, w))                -> npu_fused_matmul(x, w, "relu")
              linear/matmul without bias; the op's bias is optional, so leave it None.

    Runs at POST: in the post_grad graph linear is already decomposed and addmm's w is
    already (K, N), matching the op's x2 directly, so no transpose or reshape is inserted.

    Off by default; set TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU=1 to enable. When off it is not registered.

    Gating (strict): only A5 (Ascend 950) has the op; x1 and w must both be 2-D (x2's rank
    must equal x1's and broadcasting is unsupported); dtypes must all match and be fp16/bf16;
    a bias must be 1-D and contiguous; replaced nodes must be single-user. Otherwise the graph is left as is.
    """
    fused_op = _resolve_npu_fused_matmul()
    if fused_op is None:
        return

    changed = False
    for relu_node in list(graph.nodes):
        if not _is_relu_node(relu_node):
            continue
        if not relu_node.args or not isinstance(relu_node.args[0], torch.fx.Node):
            continue
        view_node, mm_node = _fmm_peel_view(relu_node)
        form = _fused_matmul_form(mm_node)
        if form is None:
            continue
        if len(mm_node.users) != 1:
            _fmm_reject(mm_node, f"has {len(mm_node.users)} users, expect one")
            continue

        matched = _match_fused_matmul_relu_operands(mm_node, form)
        if matched is None:
            continue
        x1, x2, bias = matched

        # bias and fused_op_type come after * in the schema and must be passed by keyword; bias
        # is optional so it is omitted when absent; in relu mode x3 must be None, also omitted.
        kwargs = {"fused_op_type": _FMM_RELU_OP_TYPE}
        if bias is not None:
            kwargs["bias"] = bias
        with graph.inserting_before(mm_node):
            fused_node = graph.call_function(
                fused_op,
                args=(x1, x2),
                kwargs=kwargs,
                name=f"{mm_node.name}_fused_relu",
            )
            # the fused result matches the replaced matmul output in shape and dtype, so reuse its meta to keep symbolic dims.
            fused_node.meta.update(mm_node.meta)

        if view_node is None:
            relu_node.replace_all_uses_with(fused_node)
        else:
            # point the original view at the fused result; its output then equals the original relu's.
            view_node.replace_input_with(mm_node, fused_node)
            relu_node.replace_all_uses_with(view_node)
        graph.erase_node(relu_node)
        changed = True
        log.info(
            "fused_matmul_relu_pass: fused relu(%s) into "
            'npu_fused_matmul(fused_op_type="relu") (form=%s, through_view=%s)',
            mm_node.name,
            form,
            view_node is not None,
        )

    eliminate_dead_code(graph, changed, fused_matmul_relu_pass.__name__)


_MSC_SLICE_TARGET = torch.ops.aten.slice.Tensor
_MSC_WHERE_TARGET = torch.ops.aten.where.self
_MSC_FULL_TARGET = torch.ops.aten.full.default
# A where condition is usually reshaped or expanded to the segment width first. Those
# are pure broadcasts, and peeling them off reveals whether it is a row mask.
_MSC_BROADCAST_TARGETS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
)


def _msc_static_last_dim(node):
    shape = get_node_shape(node, allow_symbolic=True)
    if shape is None or len(shape) != 2:
        return None
    last = shape[-1]
    return last if isinstance(last, int) else None


def _msc_bound(value, limit, default):
    """Normalize a slice start/end into a plain int within [0, limit], else None."""
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 0:
        value += limit
    return max(0, min(value, limit))


def _msc_base_limit(base, cat_dtype, rows):
    """Validate the base tensor and return its static last-dim length, else None.

    Graph inputs only: here those tables are model arguments, already materialized and
    contiguous. An intermediate buffer would be forced to materialize early by
    realize_input during lowering, possibly blocking fusion at its producer.
    """
    if not isinstance(base, torch.fx.Node) or base.op != "placeholder":
        return None
    shape = get_node_shape(base, allow_symbolic=True)
    if shape is None or len(shape) != 2:
        return None
    # Rows may be symbolic, so dynamic batch works; it just has to match the cat output.
    if not statically_known_eq(shape[0], rows):
        return None
    if get_node_dtype(base) != cat_dtype:
        return None
    return _msc_static_last_dim(base)


def _msc_plain(node, cat_dtype, rows):
    """A plain slice or a whole graph input, as (base, offset, width, limit) or None.

    Slices must be along the last dim with step 1, constant bounds and a static last dim.
    """
    if isinstance(node, torch.fx.Node) and node.op == "placeholder":
        limit = _msc_base_limit(node, cat_dtype, rows)
        return None if limit is None else (node, 0, limit, limit)

    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target is not _MSC_SLICE_TARGET or len(node.args) < 2:
        return None
    if len(node.args) > 4 and node.args[4] not in (None, 1):
        return None

    base = node.args[0]
    limit = _msc_base_limit(base, cat_dtype, rows)
    if limit is None:
        return None

    if normalize_dim(node.args[1], 2) != 1:
        return None

    start = _msc_bound(node.args[2] if len(node.args) > 2 else None, limit, 0)
    end = _msc_bound(node.args[3] if len(node.args) > 3 else None, limit, limit)
    if start is None or end is None:
        return None

    width = end - start
    # Cross-check the width against the slice's own meta in case normalization diverged.
    if width <= 0 or width != _msc_static_last_dim(node):
        return None
    return base, start, width, limit


def _msc_row_mask(node, rows):
    """Reduce a where condition to a [rows, 1] row mask, else None.

    Peels the pure broadcasts and takes the deepest qualifying node, since a shallower
    one would be materialized separately by realize_input and waste a dispatch. The
    shape must be exactly [rows, 1]: a 1D mask broadcasts along columns instead of rows,
    which means something else entirely. All four ops are the identity for [rows, 1],
    so going deeper does not change the values.
    """

    def qualifies(candidate):
        if not isinstance(candidate, torch.fx.Node):
            return False
        shape = get_node_shape(candidate, allow_symbolic=True)
        if shape is None or len(shape) != 2:
            return False
        if not statically_known_eq(shape[0], rows):
            return False
        if not statically_known_eq(shape[1], 1):
            return False
        return get_node_dtype(candidate) is torch.bool

    deepest = None
    for _ in range(5):
        if qualifies(node):
            deepest = node
        if (
            isinstance(node, torch.fx.Node)
            and node.op == "call_function"
            and node.target in _MSC_BROADCAST_TARGETS
            and node.args
        ):
            node = node.args[0]
        else:
            break
    return deepest


def _msc_is_zero_fill(node, dtype):
    """The true branch must be an all-zero constant of the segment dtype."""
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if node.target is not _MSC_FULL_TARGET or len(node.args) < 2:
        return False
    fill = node.args[1]
    if isinstance(fill, bool) or not isinstance(fill, (int, float)):
        return False
    return fill == 0 and get_node_dtype(node) == dtype


def _msc_segment(node, cat_dtype, rows):
    """Match the two segment forms, returning (base, offset, width, mask, cost).

    Either a plain slice / whole graph input, or ``where(row mask, 0, that)``.

    cost is the dispatches this segment costs today: 0 for a contiguous graph input,
    which aclnnCat reads directly, and 1 for a column slice or a masked segment. The
    saving from merging is sum(cost) - 1, which decides whether the rewrite pays off.
    """
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is _MSC_WHERE_TARGET
        and len(node.args) == 3
    ):
        cond, on_true, on_false = node.args
        if not _msc_is_zero_fill(on_true, cat_dtype):
            return None
        mask = _msc_row_mask(cond, rows)
        if mask is None:
            return None
        plain = _msc_plain(on_false, cat_dtype, rows)
        if plain is None:
            return None
        base, off, width, _ = plain
        # A differing where output width means another broadcast, not a per-row zeroing.
        if width != _msc_static_last_dim(node):
            return None
        return base, off, width, mask, 1

    plain = _msc_plain(node, cat_dtype, rows)
    if plain is None:
        return None
    base, off, width, limit = plain
    contiguous = off == 0 and width == limit
    return base, off, width, None, 0 if contiguous else 1


def _msc_runs(inputs, cat_dtype, rows, max_segments, max_sources, max_masks):
    """Split the cat inputs into runs that are adjacent and worth merging.

    Only adjacent inputs can merge: they occupy a contiguous column range in the output,
    so replacing them with one node preserves the concat order. An unrecognized input
    breaks the run.

    Segments may come from different base tensors, each costing one more kernel pointer
    argument. Source and mask counts are capped so a pathological graph cannot blow up
    the argument list.
    """
    runs = []
    cur = []       # segments accumulated so far
    start = 0      # index where cur begins in the cat argument list
    srcs = []      # base tensors used by cur, deduplicated in first-seen order
    masks = []     # row masks used by cur, likewise

    def worth(segs):
        # Merging costs one dispatch against sum(cost) today. Below that, leave it to
        # aten.cat: contiguous inputs need no extra kernel and folding them would add one.
        return len(segs) >= 2 and sum(seg[4] for seg in segs) >= 2

    for idx, inp in enumerate(inputs):
        seg = _msc_segment(inp, cat_dtype, rows)
        if seg is not None and seg[3] is not None and max_masks <= 0:
            seg = None
        if seg is None:
            if worth(cur):
                runs.append((start, cur))
            cur, srcs, masks = [], [], []
            continue

        base, mask = seg[0], seg[3]
        if cur and (
            len(cur) >= max_segments
            or (base not in srcs and len(srcs) >= max_sources)
            or (mask is not None and mask not in masks and len(masks) >= max_masks)
        ):
            # At the cap, close the run and start the next one from this segment.
            if worth(cur):
                runs.append((start, cur))
            cur, srcs, masks = [], [], []

        if not cur:
            start = idx
        cur.append(seg)
        if base not in srcs:
            srcs.append(base)
        if mask is not None and mask not in masks:
            masks.append(mask)

    if worth(cur):
        runs.append((start, cur))
    return runs


def _msc_build_node(graph, cat, run):
    """Emit one multi_slice_concat node for a run, or None if self-validation fails."""
    srcs = []
    masks = []
    for base, _, _, mask, _ in run:
        if base not in srcs:
            srcs.append(base)
        if mask is not None and mask not in masks:
            masks.append(mask)

    src_idx = [srcs.index(base) for base, _, _, _, _ in run]
    offsets = [off for _, off, _, _, _ in run]
    widths = [width for _, _, width, _, _ in run]
    mask_idx = [-1 if mask is None else masks.index(mask) for _, _, _, mask, _ in run]

    fake_srcs = [node.meta.get("val") for node in srcs]
    fake_masks = [node.meta.get("val") for node in masks]
    if any(val is None for val in fake_srcs) or any(val is None for val in fake_masks):
        return None
    # zeros_like in the reference allocates, so it has to run under the sources' own
    # fake mode; otherwise the following where raises "Mixing fake modes".
    fake_mode = getattr(fake_srcs[0], "fake_mode", None)
    try:
        # Running the reference on fake tensors yields the output meta and validates
        # that the plan is self-consistent.
        with contextlib.ExitStack() as stack:
            if fake_mode is not None:
                stack.enter_context(fake_mode)
            fake_out = _multi_slice_concat_ref(
                fake_srcs, fake_masks, src_idx, offsets, widths, mask_idx
            )
    except Exception:  # a gap in matching; skip rather than emit a wrong graph
        log.debug(
            "[inductor_fx] multi_slice_concat segment plan validation failed",
            exc_info=True,
        )
        return None

    with graph.inserting_before(cat):
        node = graph.call_function(
            MULTI_SLICE_CONCAT_TARGET,
            args=(srcs, masks, src_idx, offsets, widths, mask_idx),
        )
    node.meta["val"] = fake_out
    return node


@register_custom_pass(PassType.POST, ignore_inference_check=True)
def multi_slice_concat_pass(
    graph: torch.fx.Graph,
    *,
    max_segments: int = 64,
    max_sources: int = 32,
    max_masks: int = 4,
) -> None:
    """Fold a run of fixed-width column slices into one npu_ext::multi_slice_concat.

    aclnnCat needs contiguous inputs, so each non-contiguous slice costs a Slice copy
    and each masked one a where. The rewrite moves the same data in one dispatch, with
    fixed-width copies rather than indirect addressing.

    Matching is strict: anything that does not fit is left to aten.cat.

    max_segments caps the run length, since every segment is unrolled in the kernel
    body. max_sources caps the distinct source tensors, each costing a kernel pointer
    argument. max_masks caps the row masks held live across segments; real graphs use
    a couple per cat. The caps only bound pathological graphs.
    """
    changed = False
    for cat in list(graph.nodes):
        is_cat, dim = check_cat_op(cat)
        if not is_cat or not cat.args:
            continue
        inputs = cat.args[0]
        if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
            continue
        cat_shape = get_node_shape(cat, allow_symbolic=True)
        cat_dtype = get_node_dtype(cat)
        if cat_shape is None or cat_dtype is None or len(cat_shape) != 2:
            continue
        # Column concat along the last dim only; row concat is not this op's semantics.
        if normalize_dim(dim, 2) != 1:
            continue

        runs = _msc_runs(
            list(inputs),
            cat_dtype,
            cat_shape[0],
            max_segments,
            max_sources,
            max_masks,
        )
        if not runs:
            continue

        new_inputs = list(inputs)
        # Replace back to front so earlier run indices stay valid.
        replaced = 0
        for start, run in reversed(runs):
            node = _msc_build_node(graph, cat, run)
            if node is None:
                continue
            new_inputs[start:start + len(run)] = [node]
            replaced += 1
        if replaced == 0:
            continue

        if len(new_inputs) == 1:
            cat.replace_all_uses_with(new_inputs[0])
            graph.erase_node(cat)
        else:
            cat.args = (new_inputs,) + tuple(cat.args[1:])
        changed = True
        log.info(
            "[inductor_fx] multi_slice_concat_pass folded %d run(s) covering %d "
            "segments out of %d cat inputs",
            replaced,
            sum(len(run) for _, run in runs),
            len(inputs),
        )

    eliminate_dead_code(graph, changed, multi_slice_concat_pass.__name__)


_SSA_SUM_TARGET = torch.ops.aten.sum.dim_IntList
_SSA_STACK_TARGET = torch.ops.aten.stack.default
_SSA_ADD_TARGET = torch.ops.aten.add.Tensor
_SSA_CONVERT_TARGET = torch.ops.prims.convert_element_type.default
_SSA_UNSQUEEZE_TARGET = torch.ops.aten.unsqueeze.default
_SSA_LOW_PRECISION_FLOATS = (torch.float16, torch.bfloat16)


def _ssa_get_sum_dims(node: torch.fx.Node):
    dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if isinstance(dims, int):
        return [dims]
    if isinstance(dims, (list, tuple)) and all(isinstance(d, int) for d in dims):
        return list(dims)
    return None


def _ssa_get_keepdim(node: torch.fx.Node) -> bool:
    if len(node.args) > 2:
        return bool(node.args[2])
    return bool(node.kwargs.get("keepdim", False))


def _ssa_match_unsqueeze_cat(cat: torch.fx.Node, cat_dim):
    """Original stack form: cat([unsqueeze(x_i, d), ...], d)."""
    inputs = cat.args[0]
    cat_shape = get_node_shape(cat, allow_symbolic=True)
    if cat_shape is None:
        return None
    stack_dim = normalize_dim(cat_dim, len(cat_shape))
    if stack_dim is None:
        return None

    operands = []
    operand_shape = None
    for inp in inputs:
        if not check_unsqueeze_op(inp) or not is_single_user(inp):
            return None
        unsqueeze_dim = normalize_dim(
            inp.args[1] if len(inp.args) > 1 else 0, len(cat_shape)
        )
        if unsqueeze_dim != stack_dim:
            return None
        operand = inp.args[0]
        if not isinstance(operand, torch.fx.Node):
            return None
        shape = get_node_shape(operand, allow_symbolic=True)
        if shape is None:
            return None
        if operand_shape is None:
            operand_shape = shape
        elif not shapes_statically_equal(operand_shape, shape):
            return None
        operands.append(operand)
    return operands, stack_dim


def _ssa_match_view_cat(view: torch.fx.Node, cat: torch.fx.Node, cat_dim):
    """Simplified form: reshape(cat([x_i, ...], 0), [N, *x_shape]).

    Concatenating N equally shaped tensors along axis 0 and reshaping to [N, *x_shape]
    puts the i-th input on row i, which is exactly stack(xs, 0).
    """
    if cat_dim != 0:
        return None
    inputs = cat.args[0]
    view_shape = get_node_shape(view, allow_symbolic=True)
    if view_shape is None or len(view_shape) < 1:
        return None
    if not statically_known_eq(view_shape[0], len(inputs)):
        return None

    operands = []
    for inp in inputs:
        if not isinstance(inp, torch.fx.Node):
            return None
        shape = get_node_shape(inp, allow_symbolic=True)
        if not shapes_statically_equal(view_shape[1:], shape):
            return None
        operands.append(inp)
    return operands, 0


def _ssa_match_plain_stack(stack: torch.fx.Node):
    """Undecomposed form: aten.stack.default(xs, d)."""
    inputs = stack.args[0]
    stack_shape = get_node_shape(stack, allow_symbolic=True)
    if stack_shape is None:
        return None
    stack_dim = normalize_dim(
        stack.args[1] if len(stack.args) > 1 else stack.kwargs.get("dim", 0),
        len(stack_shape),
    )
    if stack_dim is None:
        return None
    if any(not isinstance(inp, torch.fx.Node) for inp in inputs):
        return None
    return list(inputs), stack_dim


def _ssa_match_stack_operands(sum_node: torch.fx.Node, max_inputs: int):
    """Match sum(stack(xs, d), d), returning (operands, d) or None."""
    src = sum_node.args[0] if sum_node.args else None
    if not isinstance(src, torch.fx.Node) or not is_single_user(src):
        return None

    if src.op == "call_function" and src.target == _SSA_STACK_TARGET:
        if not src.args or not isinstance(src.args[0], (list, tuple)):
            return None
        if not 2 <= len(src.args[0]) <= max_inputs:
            return None
        matched = _ssa_match_plain_stack(src)
        return _ssa_check_reduced_dim(sum_node, src, matched)

    cat = src
    view = None
    if check_view(src):
        cat = src.args[0] if src.args else None
        view = src
        if not isinstance(cat, torch.fx.Node) or not is_single_user(cat):
            return None

    is_cat, cat_dim = check_cat_op(cat)
    if not is_cat or not cat.args or not isinstance(cat.args[0], (list, tuple)):
        return None
    if not 2 <= len(cat.args[0]) <= max_inputs:
        return None

    matched = (
        _ssa_match_view_cat(view, cat, cat_dim)
        if view is not None
        else _ssa_match_unsqueeze_cat(cat, cat_dim)
    )
    return _ssa_check_reduced_dim(sum_node, src, matched)


def _ssa_check_reduced_dim(sum_node: torch.fx.Node, src: torch.fx.Node, matched):
    """The reduced axis must be the stack axis, otherwise the add chain differs."""
    if matched is None:
        return None
    operands, stack_dim = matched
    sum_dims = _ssa_get_sum_dims(sum_node)
    src_shape = get_node_shape(src, allow_symbolic=True)
    if sum_dims is None or len(sum_dims) != 1 or src_shape is None:
        return None
    if normalize_dim(sum_dims[0], len(src_shape)) != stack_dim:
        return None
    return operands, stack_dim


def _ssa_insert_cast(graph: torch.fx.Graph, node: torch.fx.Node, dtype):
    cast = graph.call_function(_SSA_CONVERT_TARGET, (node, dtype))
    propagate_fake_tensor(cast, node, lambda fake: _SSA_CONVERT_TARGET(fake, dtype))
    return cast


def _ssa_build_add_chain(graph, sum_node, operands, stack_dim):
    in_dtype = get_node_dtype(operands[0])
    if in_dtype is None or not in_dtype.is_floating_point:
        return None
    out_dtype = sum_node.kwargs.get("dtype") or get_node_dtype(sum_node) or in_dtype
    acc_dtype = torch.float32 if in_dtype in _SSA_LOW_PRECISION_FLOATS else in_dtype

    with graph.inserting_before(sum_node):
        terms = operands
        if acc_dtype != in_dtype:
            terms = [_ssa_insert_cast(graph, operand, acc_dtype) for operand in operands]

        acc = terms[0]
        for term in terms[1:]:
            add = graph.call_function(_SSA_ADD_TARGET, (acc, term))
            propagate_fake_tensor(
                add, [acc, term], lambda fakes: _SSA_ADD_TARGET(fakes[0], fakes[1])
            )
            acc = add

        if acc_dtype != out_dtype:
            acc = _ssa_insert_cast(graph, acc, out_dtype)
        if _ssa_get_keepdim(sum_node):
            unsqueeze = graph.call_function(_SSA_UNSQUEEZE_TARGET, (acc, stack_dim))
            propagate_fake_tensor(
                unsqueeze, acc, lambda fake: _SSA_UNSQUEEZE_TARGET(fake, stack_dim)
            )
            acc = unsqueeze
    return acc


@register_custom_pass(PassType.POST, ignore_inference_check=True)
def stack_sum_to_add_chain_pass(
    graph: torch.fx.Graph, *, max_add_chain_inputs: int = 64
) -> None:
    """Rewrite sum(stack(xs, d), d) into an add chain over xs.

    max_add_chain_inputs bounds the chain width: more operands mean more concurrent
    loads in one kernel, so the cap keeps register pressure in check.
    """
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != _SSA_SUM_TARGET:
            continue
        matched = _ssa_match_stack_operands(node, max_add_chain_inputs)
        if matched is None:
            continue
        operands, stack_dim = matched
        replacement = _ssa_build_add_chain(graph, node, operands, stack_dim)
        if replacement is None:
            continue
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True
        log.debug(
            "[inductor_fx] stack_sum_to_add_chain_pass folded %d-way stack into add chain",
            len(operands),
        )

    eliminate_dead_code(graph, changed, stack_sum_to_add_chain_pass.__name__)


_GMM_ADDMM_TARGET = torch.ops.aten.addmm.default
_GMM_MM_TARGET = torch.ops.aten.mm.default

# split_item=0 gives one output tensor per group, matching the separate matmuls.
# group_type=-1 means the groups share nothing and no dimension is split.
_GMM_SPLIT_ITEM_PER_GROUP = 0
_GMM_GROUP_TYPE_INDEPENDENT = -1


@dataclass
class _GmmMember:
    """One projection among the cat inputs."""

    node: torch.fx.Node
    x: torch.fx.Node
    w: torch.fx.Node
    bias: Optional[torch.fx.Node]


@functools.lru_cache(maxsize=1)
def _resolve_npu_grouped_matmul():
    """Resolve the OpOverload, or None so the pass degrades to not fusing.

    Must stay lazy: this module loads unconditionally with the pass package, so
    reading torch.ops.npu.<name> at module level would take every pass down with an
    AttributeError whenever the op is absent.
    """
    op = getattr(torch.ops.npu, "npu_grouped_matmul", None)
    op = getattr(op, "default", None)
    if op is None:
        log.warning(
            "grouped_matmul_fusion_pass disabled: "
            "torch.ops.npu.npu_grouped_matmul unavailable"
        )
    return op


def _gmm_match_projection(node: torch.fx.Node):
    """Return (x, w, bias); bias is None for a plain mm."""
    if node.op != "call_function":
        return None
    # addmm with beta/alpha is not plain x @ w + b, which the grouped op cannot express.
    if node.kwargs:
        return None
    if node.target is _GMM_ADDMM_TARGET:
        if len(node.args) != 3:
            return None
        bias, x, w = node.args
    elif node.target is _GMM_MM_TARGET:
        if len(node.args) != 2:
            return None
        bias = None
        x, w = node.args
    else:
        return None
    if not isinstance(x, torch.fx.Node) or not isinstance(w, torch.fx.Node):
        return None
    if bias is not None and not isinstance(bias, torch.fx.Node):
        return None
    return x, w, bias


def _gmm_bucket_key(
    node: torch.fx.Node, x: torch.fx.Node, w: torch.fx.Node, bias, max_rows: int
):
    """Attributes a batch must share. K may differ, which is what group_type=-1 is for.

    Symbolic dims go into the key as strings, which only pre-sorts candidates: two
    distinct symbols can print the same, so membership is confirmed separately.
    """
    out_shape = get_node_shape(node, allow_symbolic=True)
    x_shape = get_node_shape(x, allow_symbolic=True)
    w_shape = get_node_shape(w, allow_symbolic=True)
    if out_shape is None or x_shape is None or w_shape is None:
        return None
    if len(out_shape) != 2 or len(x_shape) != 2 or len(w_shape) != 2:
        return None
    # K comes from the x and w meta independently; disagreement means untrustworthy meta.
    if not statically_known_eq(x_shape[1], w_shape[0]):
        return None
    if not statically_known_eq(out_shape[0], x_shape[0]):
        return None
    if not statically_known_eq(out_shape[1], w_shape[1]):
        return None
    if bias is not None:
        bias_shape = get_node_shape(bias, allow_symbolic=True)
        if bias_shape is None or len(bias_shape) != 1:
            return None
        if not statically_known_eq(bias_shape[0], out_shape[1]):
            return None

    rows = out_shape[0]
    if not isinstance(rows, torch.SymInt) and not statically_known_leq(rows, max_rows):
        return None

    dtype = get_node_dtype(node)
    if dtype is None or get_node_dtype(x) != dtype or get_node_dtype(w) != dtype:
        return None
    if bias is not None and get_node_dtype(bias) != dtype:
        return None
    return str(rows), str(out_shape[1]), dtype, bias is not None


def _gmm_collect_buckets(
    cat: torch.fx.Node, inputs, max_rows: int
) -> List[List[_GmmMember]]:
    """Bucket the foldable cat inputs by (rows, output width, dtype, has bias).

    Members need not be adjacent in the cat: each one is replaced by its own getitem
    in place, so the concat order survives even with other inputs interleaved.
    """
    buckets = {}
    reference = {}
    seen = set()
    for inp in inputs:
        if not isinstance(inp, torch.fx.Node):
            continue
        # With other consumers the grouped call would have to move ahead of the
        # earliest one rather than the cat, and the saving is no longer clear.
        if not is_single_user(inp) or cat not in inp.users:
            continue
        # A matmul listed twice in the cat still has one user, but it can only be
        # replaced once; treating it as two members breaks on the second erase.
        if inp in seen:
            continue
        seen.add(inp)
        matched = _gmm_match_projection(inp)
        if matched is None:
            continue
        x, w, bias = matched
        key = _gmm_bucket_key(inp, x, w, bias, max_rows)
        if key is None:
            continue
        member = _GmmMember(node=inp, x=x, w=w, bias=bias)
        if key not in buckets:
            buckets[key] = []
            reference[key] = get_node_shape(inp, allow_symbolic=True)
        elif not shapes_statically_equal(
            get_node_shape(inp, allow_symbolic=True), reference[key]
        ):
            # Equal key strings do not prove equal symbols, so confirm here.
            continue
        buckets[key].append(member)
    return list(buckets.values())


def _gmm_plan_batch_size(total: int, max_groups_per_call: int) -> int:
    """Minimize the batch count first, then split evenly, leaving no short tail.

    Cutting at the cap leaves one: 80 at 32 gives 32/32/16, and that last batch pays
    a full fixed cost for little work. Even 27/27/26 measured 14% faster in wall clock.
    """
    upper = max(1, max_groups_per_call)
    batches = math.ceil(total / upper)
    return math.ceil(total / batches)


def _gmm_build_calls(
    graph: torch.fx.Graph, cat: torch.fx.Node, members, grouped_op, max_groups_per_call: int
):
    """Issue npu_grouped_matmul per batch, returning each member's getitem node."""
    size = _gmm_plan_batch_size(len(members), max_groups_per_call)
    has_bias = members[0].bias is not None
    replacements = []
    with graph.inserting_before(cat):
        for start in range(0, len(members), size):
            batch = members[start:start + size]
            kwargs = {
                "split_item": _GMM_SPLIT_ITEM_PER_GROUP,
                "group_type": _GMM_GROUP_TYPE_INDEPENDENT,
            }
            if has_bias:
                kwargs["bias"] = [m.bias for m in batch]
            grouped = graph.call_function(
                grouped_op,
                ([m.x for m in batch], [m.w for m in batch]),
                kwargs,
            )
            # Each group's output matches the matmul it replaces, so reuse the original
            # fake tensors instead of running the op again.
            grouped.meta["val"] = [m.node.meta["val"] for m in batch]
            for offset, member in enumerate(batch):
                item = graph.call_function(operator.getitem, (grouped, offset))
                item.meta["val"] = member.node.meta["val"]
                replacements.append(item)
    return replacements


@register_custom_pass(PassType.POST, ignore_inference_check=True)
def grouped_matmul_fusion_pass(
    graph: torch.fx.Graph,
    *,
    min_groups: int = 8,
    max_groups_per_call: int = 32,
    max_rows: int = 4096,
) -> None:
    """Batch the independent small GEMMs feeding one cat into npu_grouped_matmul.

    min_groups keeps wide GEMMs out: a cat over few branches tends to be the wide kind
    (QKV projections and friends), where the grouped kernel measured 12% slower than
    separate matmuls. max_groups_per_call splits large groups, since past 32 groups the
    op takes a much more expensive host path, measured 2.18 us per group jumping to
    6.90 us. max_rows is a secondary guard against GEMMs that already fill the cube; it
    only applies when the row count is static, which the batch dim usually is not, so
    min_groups carries the decision in practice.
    """
    grouped_op = _resolve_npu_grouped_matmul()
    if grouped_op is None:
        return

    changed = False
    for cat in list(graph.nodes):
        is_cat, dim = check_cat_op(cat)
        if not is_cat or not cat.args:
            continue
        inputs = cat.args[0]
        if not isinstance(inputs, (list, tuple)):
            continue
        if len(inputs) < min_groups:
            continue
        cat_shape = get_node_shape(cat, allow_symbolic=True)
        if cat_shape is None or len(cat_shape) != 2:
            continue
        # Feature-dim concat only: grouped outputs stay [M, N], which axis 0 would not match.
        if normalize_dim(dim, 2) != 1:
            continue

        for members in _gmm_collect_buckets(cat, inputs, max_rows):
            if len(members) < min_groups:
                continue
            replacements = _gmm_build_calls(
                graph, cat, members, grouped_op, max_groups_per_call
            )
            for member, replacement in zip(members, replacements):
                member.node.replace_all_uses_with(replacement)
                graph.erase_node(member.node)
            changed = True
            log.debug(
                "[inductor_fx] grouped_matmul_fusion_pass folded %d matmuls into "
                "%d npu_grouped_matmul call(s) before %s",
                len(members),
                math.ceil(len(members) / _gmm_plan_batch_size(len(members), max_groups_per_call)),
                cat,
            )

    eliminate_dead_code(graph, changed, grouped_matmul_fusion_pass.__name__)


def eliminate_dead_code(graph, changed, fn_name, POST=True):
    """Shared epilogue for all passes: if anything changed, run lint and DCE as needed and log it."""
    if changed:
        if POST:
            graph.lint()
            graph.eliminate_dead_code()
        log.info("[inductor_fx] %s works", fn_name)
