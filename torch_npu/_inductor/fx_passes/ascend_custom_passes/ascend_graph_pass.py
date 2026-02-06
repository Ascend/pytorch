import operator
import math
import torch
import torch.fx
from .register_custom_pass import register_custom_pass
from ..utils.check_op_util import (
    is_zero_like, 
    check_op_by_targets, 
    is_one_like, 
    try_match, 
    is_cast_node, 
    normalize_dtype, 
    get_node_dtype, 
    get_cast_dtype, 
    check_cat_op, 
    get_input_node, 
    get_input_kw_node, 
    check_support_op, 
    _get_tensor_meta, 
    check_view, 
    check_act_op, 
    check_squeeze_op, 
    check_unsqueeze_op, 
    check_where_op,
)
from ..utils.get_binary_fold_result import (
    get_binary_fold_result, 
    get_node_meta, 
    get_node_shape, 
    get_node_unique_id, 
    has_storage_or_layout, 
    _get_fold_result, 
    _fold_slice, 
    _fold_slice_scatter, 
    get_slice_dim, 
    get_pad_dim_and_size,
    propagate_fake_tensor,
)
from ..utils.fx_pass_level import PassType
from ...config import log


torch.library.define("npu_ext::masked_fill_inf", "(Tensor x, Tensor mask, float value) -> Tensor")


@register_custom_pass(PassType.PRE)
def cat_slice_cat_fold_pass(graph: torch.fx.Graph) -> None:
    changed = False
    for node in reversed(list(graph.nodes)):
        if node.op != 'call_function' or node.target not in (torch.cat, torch.concat):
            continue
        cat2_node = node
        cat2_inputs = cat2_node.args[0]
        cat2_dim = cat2_node.kwargs.get('dim', -1)
        cat2_shape = get_node_shape(cat2_node)
        if not cat2_shape:
            continue
        cat2_rank = len(cat2_shape)
        cat2_dim = cat2_dim + cat2_rank if cat2_dim == -1 else cat2_dim
        all_slices = True
        slice_inputs = []
        slice_ranges = []
        for inp in cat2_inputs:
            if inp.op != 'call_function' or inp.target != operator.getitem:
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
        if cat1_node.op != 'call_function' or cat1_node.target not in (torch.cat, torch.concat):
            continue
        cat1_inputs = cat1_node.args[0]
        cat1_dim = cat1_node.kwargs.get('dim', -1)
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
    # padding -> slice
    changed = False
    for node in reversed(list(graph.nodes)):
        # 检查是否为 linear 节点
        if node.op != "call_function" or node.target != torch._C._nn.pad:
            continue
        # 获取 pad 节点的输入和参数
        input_tensor = node.args[0]
        pad = node.args[1]  # 填充参数，例如 [0, 0, 0, max_seq_len]
        # value = node.args[2] if len(node.args) > 2 else 0.0  # 填充值，默认 0.0
        # 检查 pad 是否只在特定维度填充
        # 计算填充维度和填充量
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
        for user in node.users:
            if user.op == "call_function" and user.target == operator.getitem:
                start = user.args[1][pad_dim].start
                end = user.args[1][pad_dim].stop
                step = user.args[1][pad_dim].step
                slice_start = start if isinstance(start, int) else 0 # 切片起始
                slice_end = end if isinstance(end, int) else 0 # 切片结束
                slice_step = step if isinstance(step, int) else 0
                # 检查是否在维度上发生的切片，且切片范围不包含填充部分
                if (
                    slice_step != 0
                    or slice_end > input_shape[pad_dim]
                ):
                    all_slices_valid = False
                    break
                slice_nodes.append((user, (input_tensor, user.args[1])))
            else:
                all_slices_valid = False  # 非 slice 消费者，保留 pad
                break

        # 如果所有 slice 节点都满足条件，替换 pad + slice 为直接 slice
        if all_slices_valid and slice_nodes:
            for user, args in slice_nodes:
                user.args = args
            graph.erase_node(node)  # 删除 pad 节点
            changed = True
    eliminate_dead_code(graph, changed, pad_slice_fold.__name__, False)


@register_custom_pass(PassType.POST)
def fold_four_op_pass(graph: torch.fx.Graph) -> None:
    changed = False
    add_ops = (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
    sub_rsub_ops = (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar, torch.ops.aten.rsub.Tensor, torch.ops.aten.rsub.Scalar)
    mul_ops = (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
    div_ops = (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar)
    for node in reversed(graph.nodes):
        if node.op != 'call_function' or node.target not in (add_ops + sub_rsub_ops + mul_ops + div_ops):
            continue
        inp0 = node.args[0]
        inp1 = node.args[1]
        target_val = None
        is_match = False
        if check_op_by_targets(node, add_ops) or check_op_by_targets(node, sub_rsub_ops):
            is_match, target_val = try_match(inp0, inp1, is_zero_like)
        elif check_op_by_targets(node, mul_ops) or check_op_by_targets(node, div_ops):
            is_match, target_val = try_match(inp0, inp1, is_one_like)

        if is_match:
            with graph.inserting_before(node):
                fold_res = get_binary_fold_result(graph, target_val, node.meta)

            if fold_res is not None:
                node.replace_all_uses_with(fold_res)
                graph.erase_node(node)
                changed = True
    eliminate_dead_code(graph, changed, fold_four_op_pass.__name__)


@register_custom_pass(PassType.POST)
def fold_cast(graph: torch.fx.Graph) -> None:
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
                    propagate_fake_tensor(concat_node, cat_input, lambda fake: concat_node.target(fake, cat_axis))
                node.replace_all_uses_with(concat_node)
                graph.erase_node(node)
                changed = True
                flag = True
    eliminate_dead_code(graph, changed, fold_cat.__name__)


@register_custom_pass(PassType.POST)
def fold_clone(graph: torch.fx.Graph) -> None:
    changed = False
    output_node: torch.fx.Node = list(graph.nodes)[-1]
    if output_node.op != "output":
        return
    output_storages = set()
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
        org_memoryformat = inp.meta['tensor_meta'].memory_format
        target_memoryformat = (
            clone.kwargs['memory_format']
            if 'memory_format' in clone.kwargs
            else org_memoryformat
        )
        if org_memoryformat == target_memoryformat:
            clone.replace_all_uses_with(inp)
            propagate_fake_tensor(inp, clone, lambda x: x)
            graph.erase_node(clone)
            changed = True
    eliminate_dead_code(graph, changed, fold_clone.__name__)


@register_custom_pass(PassType.POST)
def fold_detach(graph: torch.fx.Graph) -> None:
    changed = False
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.detach.default
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
    changed = False
    candidates = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.expand.default
    ]

    def _same_shape(org_shape, target_shape) -> bool:
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
        input_tensor = get_node_meta(inp)
        if input_tensor is None:
            continue
        org_shape = list(input_tensor.shape)
        if _same_shape(org_shape, target_shape):
            expand.replace_all_uses_with(inp)
            propagate_fake_tensor(inp, expand, lambda x: x)
            graph.erase_node(expand)
            changed = True
    eliminate_dead_code(graph, changed, fold_expand.__name__)


@register_custom_pass(PassType.POST)
def fold_reduce(graph: torch.fx.Graph) -> None:
    changed = False
    reduce_tup = (torch.ops.aten.sum.dim_IntList,)
    candidates = [node for node in graph.nodes if node.op == "call_function" and node.target in reduce_tup]

    for reduce in reversed(candidates):
        inp = get_input_node(reduce, 0)
        shape = get_node_shape(inp)
        if shape is None:
            continue
        dims = get_input_kw_node(reduce, "dim") or list(range(len(shape)))
        if not isinstance(dims, list):
            dims = [dims]
        keep_dim = get_input_kw_node(reduce, "keepdim") or False
        if all([shape[dim] == 1 for dim in dims]):
            with graph.inserting_before(reduce):
                fold_res = _get_fold_result(graph, inp, dims, keep_dim)
            if fold_res:
                reduce.replace_all_uses_with(fold_res)
                graph.erase_node(reduce)
                changed = True
    eliminate_dead_code(graph, changed, fold_reduce.__name__)


@register_custom_pass(PassType.POST)
def fold_sink_view(graph: torch.fx.Graph) -> None:
    changed = False
    for node in reversed(graph.nodes):
        if not check_view(node):
            continue
        if len(node.users) != 1:
            continue
        user = list(node.users)[0]
        if check_act_op(user)[0]:
            with graph.inserting_before(user):
                new_act = graph.create_node(
                    op="call_function",
                    target=user.target,
                    args=(node.args[0],),
                    name=user.name + "_replacement",
                )
                propagate_fake_tensor(new_act, node.args[0], lambda fake: user.target(fake))
                new_act_view = graph.create_node(
                    op="call_function",
                    target=node.target,
                    args=(new_act, node.args[1]),
                    name=node.name + "_replacement",
                )
                propagate_fake_tensor(new_act_view, new_act, lambda fake: node.target(fake, node.args[1]))
            user.replace_all_uses_with(new_act_view)
            graph.erase_node(user)
            changed = True
        elif check_support_op(user):
            if user.args[0] is node:
                other_node = user.args[1]
            else:
                other_node = user.args[0]
            if isinstance(other_node, float) or isinstance(other_node, int):
                other_shape = []
                other_val = other_node
            else:
                other_shape = get_node_shape(other_node)
                other_val = get_node_meta(other_node)
            result_shape = get_node_shape(user)
            view_shape = get_node_shape(node)
            orig_shape = get_node_shape(node.args[0])
            if other_shape is not None and result_shape is not None and view_shape is not None and orig_shape is not None:
                no_broadcast_dims = min(len(other_shape), len(orig_shape))
                if result_shape == view_shape and (
                    len(other_shape) == 0 or orig_shape[-no_broadcast_dims:] == view_shape[-no_broadcast_dims:]
                ):
                    with graph.inserting_before(user):
                        new_add = graph.create_node(
                            op="call_function",
                            target=user.target,
                            args=(node.args[0], other_node),
                            name=user.name + "_replacement",
                        )
                        propagate_fake_tensor(new_add, node.args[0], lambda fake: user.target(fake, other_val))
                        new_add_view = graph.create_node(
                            op="call_function",
                            target=node.target,
                            args=(new_add, node.args[1]),
                            name=node.name + "_replacement",
                        )
                        propagate_fake_tensor(new_add_view, new_add, lambda fake: node.target(fake, node.args[1]))
                    user.replace_all_uses_with(new_add_view)
                    graph.erase_node(user)
                    changed = True
    eliminate_dead_code(graph, changed, fold_sink_view.__name__)


@register_custom_pass(PassType.POST)
def fold_slice(graph: torch.fx.Graph) -> None:
    changed = False
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if node.target == torch.ops.aten.slice.Tensor:
            if _fold_slice(node, graph):
                changed = True
                log.info(f"FoldSliceLike: Folded slice node {node.name}")
        elif node.target == torch.ops.aten.slice_scatter.default:
            if _fold_slice_scatter(node, graph):
                log.info(f"FoldSliceLike: Folded slice_scatter node {node.name}")
                changed = True
    eliminate_dead_code(graph, changed, fold_slice.__name__)


@register_custom_pass(PassType.POST)
def fold_squeeze(graph: torch.fx.Graph) -> None:
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


def match(a, b):
    return a == b if isinstance(b, int) else (len(b) == 1 and a in b)


@register_custom_pass(PassType.POST)
def fold_to_copy(graph: torch.fx.Graph) -> None:
    changed = False
    output_node: torch.fx.Node = list(graph.nodes)[-1]
    if output_node.op != "output":
        return
    output_storages = set()
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
        inp = copy.args[0]
        copy_meta = get_node_meta(copy)
        in_meta = get_node_meta(inp)
        if copy_meta is None or in_meta is None:
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
    changed = False
    while True:
        any_removed = False
        nodes = list(graph.nodes)
        for node in nodes:
            if node.op != "call_function":
                continue
            if node.target not in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
                continue

            view_node = node
            if not view_node.args:
                continue
            first_arg = view_node.args[0]
            if not isinstance(first_arg, torch.fx.Node):
                continue
            users = list(view_node.users)
            for user in users:
                if user.op != "call_function" or user.target != torch.ops.aten.squeeze.dim:
                    continue
                squeeze_node = user
                if not squeeze_node.args:
                    continue
                if squeeze_node.args[0] is not view_node:
                    continue
                in_meta = _get_tensor_meta(first_arg)
                squeeze_out_meta = _get_tensor_meta(squeeze_node)  
                if in_meta is None or squeeze_out_meta is None:
                    continue
                if tuple(in_meta.shape) != tuple(squeeze_out_meta.shape):
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
    int32_min, int32_max = -2**31, 2**31 - 1
    cast_dtype_limit = [torch.float32, torch.int32, torch.bool, torch.int16, torch.int8]
    changed = False
    for node in list(graph.nodes):  # 使用list避免修改时迭代问题
        if node.op == 'call_function' and node.target == torch.arange \
            and node.kwargs.get('dtype', None) == torch.int64:
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
            start = node.kwargs.get('start', start)
            end = node.kwargs.get('end', end)
            step = node.kwargs.get('step', step)
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
                num_elements = math.ceil((end - start) / step) if step > 0 else math.ceil((start - end) / -step)
                seq_min = min(start, start + (num_elements - 1) * step)
                seq_max = max(start, start + (num_elements - 1) * step)
                if seq_min > int32_min and seq_max < int32_max:
                    node.kwargs = {**node.kwargs, 'dtype': torch.int32}
                    changed = True
        if node.op == 'call_method':
            input_node = node.args[0]
            input_fake = input_node.meta.get('example_value', None) if hasattr(input_node, 'meta') else None
            input_dtype = input_fake.dtype if input_fake is not None else None
            target_dtype = node.args[1] if len(node.args) > 1 else node.kwargs.get('dtype', None)
            if input_dtype in cast_dtype_limit and node.target == 'to' and target_dtype == torch.int64:
                if len(node.args) > 1:
                    node.args = (node.args[0], torch.int32)  # 更新 positional dtype
                else:
                    node.kwargs = {**node.kwargs, 'dtype': torch.int32}  # 更新 kwargs dtype
                changed = True
    eliminate_dead_code(graph, changed, dtype_optimal_pass.__name__, False)


# for struction reason, inductor-ascend can not fully support all dual reduction case
# we add this graph pass to prevent dual reduction like x.sum()
# eg.
# x shape is [1, 1024, 2048]
# x.sum() -> x.sum(dim=2).sum(dim=1).sum(dim=0)
# todo: remove num_split patch and then remove this custom pass
@register_custom_pass(PassType.POST)
def unfold_dual_reduction_pass(graph: torch.fx.Graph) -> None:
    aten = torch.ops.aten
    changed = False
    # todo: add all reduction operator into reduction_operator_mapping
    reduction_operator_mapping = {
        "sum": aten.sum.dim_IntList,
    }

    def is_reduction(node):
        return (
            node.op == "call_function"
            and hasattr(node.target, "_opname")
            and node.target._opname in reduction_operator_mapping
        )

    candidates = [node for node in graph.nodes if is_reduction(node)]
    for reduce in candidates:
        input_tensor = get_input_node(reduce, 0)
        shape = get_node_shape(input_tensor)
        if shape is None:
            continue
        dims = get_input_kw_node(reduce, "dim") or list(range(len(shape)))
        # only for dual reduction
        if not isinstance(dims, list) or len(dims) == 1:
            continue
        keep_dim = get_input_kw_node(reduce, "keepdim") or False
        dtype = get_input_kw_node(reduce, "dtype") or None
        # unfold reduction operator from low axis to high axis
        unfold_dims = [[dim] for dim in sorted(dims, reverse=True)]
        with graph.inserting_before(reduce):
            iter_input = input_tensor
            for unfold_dim in unfold_dims:
                reduce_op = reduction_operator_mapping[reduce.target._opname]
                unfolded_reduce = graph.call_function(
                    reduce_op, args=(
                        iter_input, unfold_dim, keep_dim
                    ), kwargs={"dtype": dtype}
                )
                unfolded_reduce.meta["val"] = reduce_op(input_tensor.meta["val"], unfold_dim, keep_dim, dtype=dtype)
                iter_input = unfolded_reduce
            reduce.replace_all_uses_with(unfolded_reduce)
            graph.erase_node(reduce)
            changed = True

        eliminate_dead_code(graph, changed, unfold_dual_reduction_pass.__name__)


def eliminate_dead_code(graph, changed, fn_name, POST=True):
    if changed:
        if POST:
            graph.lint()
            graph.eliminate_dead_code()
        log.info(f"[inductor_fx] {fn_name} works")