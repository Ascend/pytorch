import math
import operator

import torch
import torch.fx
from torch.utils._ordered_set import OrderedSet

from ...config import log
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
    is_zero_like,
    match,
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
from .register_custom_pass import register_custom_pass


torch.library.define(
    "npu_ext::masked_fill_inf", "(Tensor x, Tensor mask, float value) -> Tensor"
)


@register_custom_pass(PassType.PRE)
def cat_slice_cat_fold_pass(graph: torch.fx.Graph) -> None:
    """折叠 cat -> slice -> cat 的冗余模式：当后一个 cat 的输入是前一个 cat 的连续切片
    且切片范围完整覆盖原 cat 时，直接复用前一个 cat 的结果。"""
    changed = False
    for node in reversed(list(graph.nodes)):
        if node.op != "call_function" or node.target not in (torch.cat, torch.concat):
            continue
        cat2_node = node
        cat2_inputs = cat2_node.args[0]
        cat2_dim = cat2_node.kwargs.get("dim", -1)
        cat2_shape = get_node_shape(cat2_node)
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
        cat1_shape = get_node_shape(cat1_node)
        if not cat1_shape:
            continue
        cat1_rank = len(cat1_shape)
        cat1_dim = cat1_dim + cat1_rank if cat1_dim == -1 else cat1_dim
        if cat1_dim != cat2_dim or cat1_shape != cat2_shape:
            continue
        sorted_ranges = [
            (sl.start, sl.stop, sl.step)
            for sl in slice_ranges
            if isinstance(sl.start, int) and isinstance(sl.stop, int)
        ]
        ranges_match = True
        expected_start = 0
        for start, stop, step in sorted_ranges:
            if start != expected_start or step not in (1, None):
                ranges_match = False
                break
            expected_start = stop
        ranges_match = ranges_match and len(sorted_ranges) == len(cat1_inputs)

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
    """折叠 pad -> slice 模式：当切片范围位于 pad 前的有效数据区域内时，
    直接基于原输入做 slice，省去 pad 节点。"""
    # padding -> slice
    changed = False
    for node in reversed(list(graph.nodes)):
        # 检查是否为 linear 节点
        if node.op != "call_function" or node.target != torch._C._nn.pad:
            continue
        # 获取 pad 节点的输入和参数
        input_tensor = node.args[0]
        pad = node.args[1]
        input_shape = get_node_shape(input_tensor)
        if input_shape is None:
            continue
        pad_dim, _ = get_pad_dim_and_size(pad, input_shape)
        if pad_dim is None:
            continue
        # 查找 pad 节点的消费者（后续节点）
        # 检查所有下游 slice 节点
        all_slices_valid = True
        slice_nodes = []
        for user in list(node.users):
            if user.op != "call_function" or user.target != operator.getitem:
                all_slices_valid = False
                break
            # 取出索引 tuple
            idx = user.args[1]
            if not isinstance(idx, (tuple, list)) or len(idx) <= pad_dim:
                all_slices_valid = False
                break
            start = idx[pad_dim].start
            end = idx[pad_dim].stop
            step = idx[pad_dim].step
            slice_start = start if isinstance(start, int) else 0
            slice_end = end if isinstance(end, int) else None
            slice_step = step if isinstance(step, int) else 1
            # 检查是否在维度上发生的切片，且切片范围不包含填充部分
            is_valid_prefix = (
                isinstance(slice_end, int)
                and slice_end <= input_shape[pad_dim]
                and slice_step in (1, None)
                and slice_start <= slice_end
            )
            if not is_valid_prefix:
                all_slices_valid = False
                break
            slice_nodes.append((user, (input_tensor, idx)))

        # 如果所有 slice 节点都满足条件，替换 pad + slice 为直接 slice
        if all_slices_valid and slice_nodes:
            for user, new_args in slice_nodes:
                user.args = new_args
            graph.erase_node(node)  # 删除 pad 节点
            changed = True
    eliminate_dead_code(graph, changed, pad_slice_fold.__name__, False)


@register_custom_pass(PassType.POST)
def fold_four_op_pass(graph: torch.fx.Graph) -> None:
    """消除四则运算中的恒等操作：如 x+0、x-0、0-x、x*1、x/1 等，
    直接用非零/非一侧的输入替换整个二元运算节点。"""
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
    """消除恒等 cast：当 cast 的目标 dtype 与输入 dtype 相同时，
    直接用输入替换 cast 节点。"""
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
    """合并嵌套 cat：当某个 cat 的输入也是同维度的 cat 且只被使用一次时，
    将内层 cat 的输入展平到外层，减少一次拼接开销。"""
    changed = False
    flag = True
    while flag:
        flag = False
        for node in list(graph.nodes):
            is_cat, cat_axis = check_cat_op(node)
            if not is_cat:
                continue
            node_shape = get_node_shape(node)
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
                        inp_shape = get_node_shape(inp)
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
    """消除非输出且 memory_format 不变的 clone：当 clone 不影响存储语义且其结果
    并非图输出时，直接用输入替换。"""
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
    """消除推理图中的 detach 节点：detach 在前向不影响数值结果，
    可直接用其输入替换。"""
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
    """消除恒等 expand：当目标 shape 与输入 shape 一致（-1 视为相同）时，
    直接用输入替换 expand 节点。"""
    changed = False
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.expand.default
    ]

    def _same_shape(org_shape, target_shape) -> bool:
        """判断两个 shape 是否等价（目标 shape 中的 -1 视为与原 shape 同维度）。"""
        if len(org_shape) != len(target_shape):
            return False
        for os, ts in zip(org_shape, target_shape):
            if os != ts and ts != -1:
                return False
        return True

    for expand in candidates:
        inp = expand.args[0]
        target_shape = expand.args[1]
        if not isinstance(target_shape, list):
            continue
        inp_shape = get_node_shape(inp)
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
    """消除作用在 size 为 1 的维度上的 reduce（如 sum）：这种 reduce 不改变数值，
    可直接用对应的视图操作替换。"""
    changed = False
    reduce_tup = (torch.ops.aten.sum.dim_IntList,)
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target in reduce_tup
    ]

    for reduce in reversed(candidates):
        inp = get_input_node(reduce, 0)
        shape = get_node_shape(inp)
        if shape is None:
            continue
        dims = get_input_kw_node(reduce, "dim") or list(range(len(shape)))
        if not isinstance(dims, list):
            dims = [dims]
        keep_dim = get_input_kw_node(reduce, "keepdim") or False
        if all(shape[dim] == 1 for dim in dims):
            with graph.inserting_before(reduce):
                fold_res = _get_fold_result(graph, inp, dims, keep_dim)
            if fold_res:
                reduce.replace_all_uses_with(fold_res)
                graph.erase_node(reduce)
                changed = True
    eliminate_dead_code(graph, changed, fold_reduce.__name__)


@register_custom_pass(PassType.POST)
def fold_sink_view(graph: torch.fx.Graph) -> None:
    """将 view 操作下沉到其后续的激活/逐元素算子之后：先在原 shape 上执行计算，
    再做 view，从而便于后续融合且不影响数值结果。"""
    changed = False
    for node in reversed(graph.nodes):
        if not check_view(node):
            continue
        if len(node.users) != 1:
            continue
        view_shape = get_node_shape(node)
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
                    new_act_view, new_act, lambda fake: node.target(fake, node.args[1])
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
                other_shape = get_node_shape(other_node)
                other_val = get_node_meta(other_node)
            result_shape = get_node_shape(user)
            orig_shape = get_node_shape(node.args[0])
            if (
                other_shape is not None
                and result_shape is not None
                and view_shape is not None
                and orig_shape is not None
            ):
                no_broadcast_dims = min(len(other_shape), len(orig_shape))
                if result_shape == view_shape and (
                    len(other_shape) == 0
                    or orig_shape[-no_broadcast_dims:]
                    == view_shape[-no_broadcast_dims:]
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
                            lambda fake: node.target(fake, node.args[1]),
                        )
                    user.replace_all_uses_with(new_add_view)
                    graph.erase_node(user)
                    changed = True
    eliminate_dead_code(graph, changed, fold_sink_view.__name__)


@register_custom_pass(PassType.POST)
def fold_slice(graph: torch.fx.Graph) -> None:
    """折叠无效的 slice / slice_scatter：当切片范围等同于完整范围时，
    用输入直接替换以消除冗余切片。"""
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
    """合并相邻的 squeeze/unsqueeze：处理 squeeze→squeeze 串联以及
    squeeze→unsqueeze 互逆这两类冗余形变。"""
    changed = False
    for node in reversed(graph.nodes):
        if not check_squeeze_op(node):
            continue
        prev = node.args[0]
        if len(prev.users) > 1:
            continue
        # case1: squeeze → squeeze
        if check_squeeze_op(prev):
            if len(node.args) == 1:
                node.replace_input_with(prev, prev.args[0])
                changed = True
            elif len(prev.args) == 1:
                node.replace_all_uses_with(prev)
                propagate_fake_tensor(prev, node, lambda x: x)
                graph.erase_node(node)
                changed = True
        # case2: squeeze → unsqueeze
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
    """消除无副作用的 _to_copy：当 dtype/device/memory_format 等都未发生改变，
    且其结果非图输出时，直接用输入替换。"""
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
        """判断一个 _to_copy 节点是否为无效拷贝：所有可观察属性
        （dtype、device、layout、memory_format 等）与输入完全一致。"""
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
    """折叠连续的 view 类操作：将多个 view/reshape/squeeze/unsqueeze 链合并为一次形变，
    同时消除目标 shape 与输入 shape 相同的恒等 view。"""
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
            if not isinstance(target_shape, list):
                continue
            inp_shape = get_node_shape(inp)
            if inp_shape is not None:
                if target_shape == list(inp_shape):
                    view.replace_all_uses_with(inp)
                    propagate_fake_tensor(inp, view, lambda x: x)
                    graph.erase_node(view)
                    changed = True
    eliminate_dead_code(graph, changed, view_fold_pass.__name__)


@register_custom_pass(PassType.POST)
def fold_where(graph: torch.fx.Graph) -> None:
    """折叠 where 中两个分支恒等的情况：当 true / false 分支取值相同（或同为全 0/全 1）时，
    用其中一支直接替换 where 节点。"""
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
    """消除冗余的 view→squeeze 组合：当 view 后的 squeeze 输出 shape/dtype 与
    view 输入完全一致时，整段链路可直接被 view 的输入替换。"""
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
                in_shape = get_node_shape(first_arg)
                squeeze_out_shape = get_node_shape(squeeze_node)
                if in_meta is None or squeeze_out_meta is None:
                    continue
                if in_shape is None or squeeze_out_shape is None:
                    continue
                if in_shape != squeeze_out_shape:
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
    """将不必要的 int64 优化为 int32：若 torch.arange 或 to(int64) 的取值
    可被 int32 安全表示，则降级 dtype 以减少访存与计算开销。"""
    int32_min, int32_max = -(2**31), 2**31 - 1
    cast_dtype_limit = [torch.float32, torch.int32, torch.bool, torch.int16, torch.int8]
    changed = False
    for node in list(graph.nodes):  # 使用list避免修改时迭代问题
        if (
            node.op == "call_function"
            and node.target == torch.arange
            and node.kwargs.get("dtype", None) == torch.int64
        ):
            # 步骤1: 动态提取 start/end/step (处理不同 args 长度)
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
            # 合并 kwargs 覆盖 (e.g., 用户指定 kwargs['start'])
            start = node.kwargs.get("start", start)
            end = node.kwargs.get("end", end)
            step = node.kwargs.get("step", step)
            # 如果 end 为 None，假设无限或跳过 (罕见，但安全)
            if end is None:
                continue
            # 静态范围检查 (所有参数是常量)
            if all(isinstance(p, (int, float)) for p in [start, step, end]):
                if step == 0:
                    continue
                # 如果 step 非整数且 dtype 是 int，警告 (arange 会自动转为 float)
                if not isinstance(step, int):
                    continue
                # 计算序列长度和 min/max 值
                num_elements = (
                    math.ceil((end - start) / step)
                    if step > 0
                    else math.ceil((start - end) / -step)
                )
                seq_min = min(start, start + (num_elements - 1) * step)
                seq_max = max(start, start + (num_elements - 1) * step)
                if seq_min > int32_min and seq_max < int32_max:
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
                    node.args = (node.args[0], torch.int32)  # 更新 positional dtype
                else:
                    node.kwargs = {
                        **node.kwargs,
                        "dtype": torch.int32,
                    }  # 更新 kwargs dtype
                changed = True
    eliminate_dead_code(graph, changed, dtype_optimal_pass.__name__, False)


@register_custom_pass(PassType.PRE)
def fusion_attention_v3_pass(graph: torch.fx.Graph) -> None:
    """将 npu_fusion_attention 升级为 v3 版本：以等价的 v3 算子替换原节点，
    保留全部参数与元数据，从而启用更高性能的实现。"""
    changed = False
    for node in list(graph.nodes):  # 使用list避免迭代时修改图结构
        if (
            node.op == "call_function"
            and node.target == torch.ops.npu.npu_fusion_attention.default
        ):
            # 创建新节点调用v3版本
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
    """将拼接来自同一父张量切片的 cat 转换为 view 或 roll：当多个 slice 完整覆盖
    某一维度时，cat 等价于恒等视图或循环位移，从而避免实际数据搬运。"""
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
        cat_shape = get_node_shape(cat)
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
            if not isinstance(sl_start, int):
                valid = False
                break
            if sl_end is not None and not isinstance(sl_end, int):
                valid = False
                break
            if parent is None:
                parent = p
            elif parent is not p:
                valid = False
                break
            intervals.append((sl_start, sl_end))

        if not valid or parent is None:
            continue
        parent_shape = get_node_shape(parent)
        if parent_shape is None or len(parent_shape) != rank:
            continue
        if list(parent_shape) != list(cat_shape):
            continue
        dim_size_raw = parent_shape[cat_dim]
        if isinstance(dim_size_raw, torch.SymInt):
            continue
        dim_size = int(dim_size_raw)

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
                "→ identity view (full cover [0, %d))",
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
            "→ roll(shift=%d) (cyclic rotation of full cover [0, %d))",
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
    """将仅用于广播的 repeat 改写为 expand：在所有使用者都支持广播的前提下，
    用零拷贝的 expand 替代物理拷贝的 repeat。"""
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

        in_shape = get_node_shape(inp)
        if in_shape is None:
            continue
        if len(repeats) != len(in_shape):
            continue

        valid = True
        for r, s in zip(repeats, in_shape):
            if isinstance(r, torch.SymInt) or isinstance(s, torch.SymInt):
                valid = False
                break
            if not (isinstance(r, int) and isinstance(s, int)):
                valid = False
                break
            if r != 1 and s != 1:
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

        out_shape = [int(r) * int(s) for r, s in zip(repeats, in_shape)]

        inp_fake = inp.meta.get("val")
        fake_mode = (
            getattr(inp_fake, "fake_mode", None) if inp_fake is not None else None
        )

        with graph.inserting_before(rpt):
            exp = graph.call_function(
                torch.ops.aten.expand.default,
                args=(inp, list(out_shape)),
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
            "repeat_to_expand_pass: rewrote repeat(%s, %s) → "
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


def _prims_iota_value_range(node):
    """计算 prims.iota 节点产生序列的 [lo, hi) 取值范围；
    若参数非常量整数则返回 None。"""
    if not (
        node.op == "call_function"
        and node.target is torch.ops.prims.iota.default
        and node.args
    ):
        return None
    length = node.args[0]
    start = node.kwargs.get("start", 0)
    step = node.kwargs.get("step", 1)
    if not (
        isinstance(length, int) and isinstance(start, int) and isinstance(step, int)
    ):
        return None
    if length <= 0:
        return (start, start)
    last = start + (length - 1) * step
    lo = min(start, last)
    hi = max(start, last)
    return (lo, hi + 1)


def _collect_iota_downcast_closure(iota_node):
    """收集从 iota 出发、只经过 dtype 透明算子直至遇到收尾算子的所有中间节点；
    若遇到不支持的算子则返回 None 表示无法降级。"""
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


def _refresh_fake_meta(node, fake_mode):
    """基于当前 args/kwargs 在 fake_mode 下重新执行算子，刷新节点的 meta['val'] FakeTensor。"""

    def resolve(arg):
        if isinstance(arg, torch.fx.Node):
            return arg.meta.get("val", arg)
        if isinstance(arg, (list, tuple)):
            return type(arg)(resolve(x) for x in arg)
        return arg

    try:
        with fake_mode:
            new_val = node.target(
                *[resolve(a) for a in node.args],
                **{k: resolve(v) for k, v in node.kwargs.items()},
            )
        node.meta["val"] = new_val
    except Exception:
        pass


_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _hashable_const_key(value):
    """将常量参数（含嵌套 list/tuple/dict）转为可哈希的 key，便于做常量折叠的 CSE 比较。"""
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


def _cse_constant_call(graph, target):
    """对指定 target 的常量参数调用做公共子表达式消除：
    相同 args/kwargs 的重复调用只保留首次，其余复用结果。"""
    seen = {}
    changed = False
    for n in list(graph.nodes):
        if n.op != "call_function" or n.target is not target:
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
    """对 iota/arange/full 做常量 CSE，并尝试将取值范围在 int32 内的 int64 iota
    降级为 int32；同时将 cmp(sub(a,b),0) 简化为 cmp(a,b)。"""
    changed = False

    iota_target = torch.ops.prims.iota.default
    for tgt in (
        iota_target,
        torch.ops.aten.arange.default,
        torch.ops.aten.full.default,
    ):
        if _cse_constant_call(graph, tgt):
            changed = True

    for iota in list(graph.nodes):
        if iota.op != "call_function" or iota.target is not iota_target:
            continue
        if iota.kwargs.get("dtype") is not torch.int64:
            continue
        rng = _prims_iota_value_range(iota)
        if rng is None:
            continue
        lo, hi_exc = rng
        if lo < _INT32_MIN or hi_exc - 1 > _INT32_MAX:
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
        _refresh_fake_meta(iota, fake_mode)
        if iota.meta.get("val") is None or iota.meta["val"].dtype is not torch.int32:
            new_kwargs["dtype"] = torch.int64
            iota.kwargs = new_kwargs
            _refresh_fake_meta(iota, fake_mode)
            continue

        middle_nodes_in_topo = [n for n in graph.nodes if id(n) in middle_ids]
        for n in middle_nodes_in_topo:
            _refresh_fake_meta(n, fake_mode)
        for n in list(graph.nodes):
            if (
                n.op == "call_function"
                and n.target in _IOTA_DTYPE_CLOSING_OPS
                and any(u is iota or id(u) in middle_ids for u in n.all_input_nodes)
            ):
                _refresh_fake_meta(n, fake_mode)

        changed = True
        log.info(
            "fold_iota_arithmetic_pass: downcast iota[%d,%d) int64 → int32"
            " (%d transparent user%s in closure)",
            lo,
            hi_exc,
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
        new_target = cmp_scalar_to_tensor[cmp.target]
        with graph.inserting_before(cmp):
            new_cmp = graph.call_function(new_target, args=(a, b))
        if "val" in cmp.meta:
            new_cmp.meta["val"] = cmp.meta["val"]
        cmp.replace_all_uses_with(new_cmp)
        changed = True
        log.info(
            "fold_iota_arithmetic_pass: folded %s(sub(a, b), 0) → %s(a, b)",
            cmp.target,
            new_target,
        )

    eliminate_dead_code(graph, changed, fold_iota_arithmetic_pass.__name__)


def _extract_const_full_scalar(node):
    """从 aten.full 节点中提取其常量填充标量；
    若节点不是 full 或填充值非标量则返回 None。"""
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
    """压缩 cast(where(bool_mask, full(c1), full(c2))) 模式：当两路常量构成 0/1 选择时，
    用 mask 自身（或 logical_not(mask)）替代该 where+cast，消除显式广播。"""
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
            action = "rewire cast input from where → mask"

        changed = True

        log.info(
            "broadcast_const_mask_compress: collapsed "
            "cast[%s](where(bool_mask, full(%s), full(%s))) → %s "
            "(%s; dropping explicit broadcast to %s)",
            target_dtype,
            t_val,
            f_val,
            replacement_kind,
            action,
            get_node_shape(w),
        )

    eliminate_dead_code(graph, changed, broadcast_const_mask_compress.__name__)


def _is_zero_tensor_source(node):
    """判断节点是否为「全 0 张量来源」：标量 0，或 zeros / zeros_like / full(0) 调用。"""
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
    """剥离最外层的逻辑/按位取反算子，返回 (内部节点, 是否被取反)；
    若不是取反则原样返回并标记为 False。"""
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
    """判断两个 mask 是否恰好互为逻辑取反：底层节点相同，且仅有一边被 not 包裹。"""
    if m1 is m2:
        return False
    s1, neg1 = _strip_logical_not(m1)
    s2, neg2 = _strip_logical_not(m2)
    return s1 is s2 and (neg1 ^ neg2)


def _match_masked_zero_where(node):
    """匹配形如 where(mask, val, 0) 的模式：返回 (mask, val)，否则返回 None。"""
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
    """将 where(m, a, 0) + where(~m, b, 0) 合成单个 where(m, a, b)：
    两个互补的掩码加法等价于一次三目选择，可节省一次加法与一次 where。"""
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
            "where(m, a, 0) + where(~m, b, 0) → where(m, a, b) "
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
    """从 node 出发反向走单一使用者的 view 类链，直到遇到 cast 节点：
    返回 (中间 view 链, cast 节点)，匹配失败则返回 (None, None)。"""
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
    """以 base_node 为起点，按相同顺序重放给定的 view 链，
    在图中创建对等的新节点链并刷新其 fake meta。"""
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
    """将 cast[dtype](bool_mask) * x 改写为 where(bool_mask, x, 0)：
    避免显式的 bool→numeric 类型转换与广播乘法，让后端有更多融合机会。"""
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
        chain_desc = " → ".join(v.target.__name__ for v in chain) if chain else "direct"
        log.info(
            "bool_cast_mul_to_where_pass: folded "
            "cast[%s](bool_mask) * x → where(bool_mask, x, 0) "
            "(mask=%s, view-chain=[%s], shape=%s)",
            cast_target_dtype,
            cast_src.name,
            chain_desc,
            get_node_shape(other),
        )

    eliminate_dead_code(graph, changed, bool_cast_mul_to_where_pass.__name__)


def _peel_single_user_relu_sign(node):
    """匹配并剥离单使用者的 relu(sign(x)) 串联结构，返回内部的 x；不匹配时返回 None。"""
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
    """将 sum(abs(relu(sign(x)) - relu(sign(y)))) 融合为符号位的汉明距离：
    用 sum(ne(gt(x,0), gt(y,0))) 等价替换，简化算子链并降低中间张量成本。"""
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
                _refresh_fake_meta(n, fake_mode)
        if "val" not in new_sum.meta and "val" in sum_node.meta:
            new_sum.meta["val"] = sum_node.meta["val"]

        sum_node.replace_all_uses_with(new_sum)
        changed = True
        log.info(
            "sign_diff_hamming_fuse_pass: folded "
            "sum(abs(sub(relu(sign(%s)), relu(sign(%s))))) → "
            "sum(ne(gt(·,0), gt(·,0)), dtype=%s) "
            "(dim=%s, keepdim=%s)",
            x_src.name if isinstance(x_src, torch.fx.Node) else x_src,
            y_src.name if isinstance(y_src, torch.fx.Node) else y_src,
            out_dtype,
            dim_arg,
            keepdim_arg,
        )

    eliminate_dead_code(graph, changed, sign_diff_hamming_fuse_pass.__name__)


def _has_default_embedding_args(node):
    """判断 aten.embedding 调用是否使用全部默认参数
    （padding_idx=-1、未启用 scale_grad_by_freq 与 sparse）。"""
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
    """生成可哈希的 shape key：SymInt 维度转字符串，其它维度转 int，便于 embedding 分组。"""
    return tuple(str(d) if isinstance(d, torch.SymInt) else int(d) for d in shape)


def _weight_node_key(w):
    """生成 embedding 权重节点的标识 key：参数/常量按 (op, target) 共享，其余按 id 区分。"""
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
    """根据 reduce 算子签名，构造其规约维度参数：dim_int 版本传单个 int，dim_list 版本传列表。"""
    if target in _REDUCE_OPS_DIM_INT:
        return (input_node, reduce_dim)
    return (input_node, [reduce_dim])


def _detect_reduce_pattern(nodes, cat_dim):
    """检查每个 embedding 节点的唯一使用者是否都是同一种沿 cat_dim 规约且 keepdim=False 的 reduce；
    若一致则返回 (reduce_target, reduce_nodes)，否则返回 None。"""
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
    """检测一组 embedding 节点的 indices 是否来自同一父张量的同维度连续 slice，
    并且这些 slice 完整无重叠地覆盖该维度；满足时返回 (parent, slice_dim)。"""
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

    parent_shape = get_node_shape(parent)
    if parent_shape is None or slice_dim >= len(parent_shape):
        return None, None
    dim_size = parent_shape[slice_dim]
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
    """对一组共享权重的 embedding 调用进行批量融合：校验索引来源与下游 reduce 模式后，
    转发到 Pattern C 的 reshape-first 实现去合并为单个 embedding+reduce。"""
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
    """尝试将「N 个 select + 同一 cat」的下游链折叠成一次 reshape：
    当所有 select 在 cat 的末维按序连续出现时，可用 flatten 的 reshape 直接接入 cat。"""
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
    """实现 Pattern C 的批量 embedding 融合：先对父 indices reshape 出额外的 N 维，
    再做一次 embedding + reduce，最后用 select（或 cat 折叠）接回原下游使用者。"""
    N = len(nodes)

    parent_shape = get_node_shape(parent_node)
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
        reshaped_parent = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(parent_node, list(new_idx_shape)),
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
            "(reshape-first + cat-collapse) — replaced %d "
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
            "— replaced %d (slice+emb+reduce) with "
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
    """批量融合 embedding 调用：按权重和索引 shape 分组，对同组的多次 embedding+reduce
    合并为单次 reshape→embedding→reduce，从而显著降低调度与访存开销。"""
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

        w_shape = get_node_shape(weight)
        idx_shape = get_node_shape(indices)
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


def eliminate_dead_code(graph, changed, fn_name, POST=True):
    """所有 pass 共用的收尾工具：如本次产生过改动，则按需执行 lint 与死代码消除并记录日志。"""
    if changed:
        if POST:
            graph.lint()
            graph.eliminate_dead_code()
        log.info("[inductor_fx] %s works", fn_name)
