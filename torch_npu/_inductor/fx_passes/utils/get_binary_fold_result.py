from typing import Any, Dict, Optional, Union, List
import torch
from torch import fx
from torch.multiprocessing.reductions import StorageWeakRef


MAX_INT64 = 9223372036854775807


def get_node_meta(node: torch.fx.Node):
    if (node.meta == {}) or ('val' not in node.meta and 'tensor_meta' not in node.meta):
        return None
    meta_info = node.meta
    return get_val_meta_info(meta_info)


def get_val_meta_info(target_meta: Dict[str, Any]):
    if 'val' in target_meta:
        return target_meta['val']
    if 'tensor_meta' in target_meta:
        return target_meta['tensor_meta']
    return None


def get_node_shape(node: torch.fx.Node):
    if (node.meta == {}) or ('example_value' not in node.meta and 'tensor_meta' not in node.meta and 'val' not in node.meta):
        return None
    shape = None
    if 'val' in node.meta:
        shape = node.meta['val'].shape
    elif 'example_value' in node.meta:
        example_value = node.meta.get('example_value')
        if hasattr(example_value, 'size'):
            shape = example_value.size()
    elif 'tensor_meta' in node.meta:
        shape = node.meta['tensor_meta'].shape
    if any(isinstance(s, torch.SymInt) for s in shape):
        return None
    return shape


def get_node_unique_id(node: torch.fx.Node):
    if 'val' in node.meta:
        val = node.meta['val']
        if isinstance(val, torch.Tensor):
            return StorageWeakRef(val.untyped_storage())
    if 'tensor_meta' in node.meta:
        tm = node.meta.get('tensor_meta')
        if hasattr(tm, 'dtype') and hasattr(tm, 'stride') and hasattr(tm, 'shape'):
            return (
                "raw_meta",
                tm.dtype,
                tuple(tm.shape),
                tuple(tm.stride),
                node.name,
                node.target,
                node.args
            )
    return None


def has_storage_or_layout(node):
    return get_node_unique_id(node) is not None


def get_binary_fold_result(
    graph: torch.fx.Graph, inp: Union[int, float, fx.Node], target_meta: Dict[str, Any]
) -> Optional[fx.Node]:
    scalar_tup = (
        int,
        float
    )
    if isinstance(inp, fx.Node):
        val_meta = get_val_meta_info(target_meta)
        node_meta = get_node_meta(inp)
        node_meta_shape = get_node_shape(inp)
        if node_meta_shape is None:
            return None
        if val_meta is None:
            return None
        if node_meta is None:
            return None
        if type(inp) in scalar_tup:
            if any(isinstance(s, torch.SymInt) for s in val_meta.shape):
                return None
            new_node = graph.call_function(
                torch.ops.aten.full.default,
                args=(val_meta.shape, inp),
                kwargs={
                    "dtype": val_meta.dtype,
                    "device": target_meta.device
                }
            )
            propagate_fake_tensor(new_node, inp, lambda fake: fake.full(val_meta.shape))
        else:
            if node_meta_shape == val_meta.shape:
                expand = inp
            else:
                if any(isinstance(s, torch.SymInt) for s in val_meta.shape):
                    return None
                expand = graph.call_function(
                    torch.ops.aten.expand.default,
                    args=(
                        inp,
                        val_meta.shape
                    )
                )
                propagate_fake_tensor(expand, inp, lambda fake: fake.expand(val_meta.shape))
            if node_meta.dtype != val_meta.dtype:
                copy = graph.call_function(
                    torch.ops.prims.convert_element_type.default,
                    args=(expand,),
                    kwargs={
                        "dtype": val_meta.dtype
                    }
                )
                propagate_fake_tensor(copy, expand, lambda fake: fake.to(dtype=val_meta.dtype))
            else:
                copy = graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(expand,),
                )
                propagate_fake_tensor(copy, expand, lambda fake: fake.clone())
            return copy
    return None


def _get_fold_result(graph: torch.fx.Graph, src, dims: List[int], keep_dim: bool) -> fx.Node:
    # 检查 src 是否被原地操作引用
    for user in src.users:
        if user.op == "call_function" and user.target in [torch.ops.aten.add_.Tensor, torch.ops.aten.add_.Scalar, torch.ops.aten.copy_.default]:
            return None
    if keep_dim:
        return src
    else:
        view = graph.call_function(
            torch.ops.aten.squeeze.dims,
            args=(
                src,
                dims,
            ),
        )
        propagate_fake_tensor(view, src, lambda fake: fake.squeeze(*dims))
        return view


def propagate_fake_tensor(node, input_node, simulate_fn):
    if 'val' in node.meta:
        return

    if isinstance(input_node, (list, tuple)):
        fake_inputs = []

        for inp in input_node:
            if 'val' in inp.meta:
                fake_inputs.append(inp.meta['val'])
        if len(fake_inputs) == 0:
            return
        fake_out = simulate_fn(fake_inputs)
        if fake_out is None:
            return
        node.meta['val'] = fake_out
        return
    if 'val' in input_node.meta:
        fake_in = input_node.meta['val']
        fake_out = simulate_fn(fake_in)
        if fake_out is None:
            return
        node.meta['val'] = fake_out
        return


def _fold_slice(node: torch.fx.Node, graph: torch.fx.Graph) -> bool:
    if len(node.args) < 4:
        return False
    src_node, dim, start, end = node.args[0], node.args[1], node.args[2], node.args[3]
    if not isinstance(dim, int) or not isinstance(start, int):
        return False
    if start != 0:
        return False
    if end is not None and not isinstance(end, int):
        return False
    src_shape = get_node_shape(src_node)
    if src_shape is None:
        return False

    node_shape = get_node_shape(node)
    if node_shape is None:
        return False

    if src_shape != node_shape:
        return False

    if dim >= len(src_shape):
        return False

    dim_length = src_shape[dim]

    is_full_slice = (
        isinstance(start, int) and start == 0 and
        (end is None or end >= MAX_INT64 or end >= dim_length)
    )

    if is_full_slice:
        node.replace_all_uses_with(src_node)
        propagate_fake_tensor(src_node, node, lambda x: x)
        graph.erase_node(node)
        return True
    return False


def _fold_slice_scatter(node: torch.fx.Node, graph: torch.fx.Graph) -> bool:
    if len(node.args) < 5:
        return False
    base_node, view_node, dim, start, end = node.args[:5]

    if not isinstance(dim, int) or not isinstance(start, int) or not isinstance(end, int):
        return False
    base_shape = get_node_shape(base_node)
    view_shape = get_node_shape(view_node)
    if (base_shape is None) or (view_shape is None):
        return False

    if base_shape != view_shape:
        return False

    if start != 0:
        return False

    dim_length = view_shape[dim]
    if end != dim_length:
        return False

    node.replace_all_uses_with(view_node)
    propagate_fake_tensor(view_node, node, lambda x: x)
    graph.erase_node(node)
    return True


def get_pad_dim_and_size(pad, input_shape):
    """
    从 pad 参数和输入张量形状中提取填充的维度和填充量。
    Args:
        pad: 填充参数列表，例如 [0, 0, 0, max_seq_len]。
        input_shape: 输入张量的形状，例如 [128, 50, 128]。
    Returns:
        pad_dim: 填充的维度索引（从 0 开始）。
        pad_size: 右侧填充量。
    """
    N = len(input_shape)  # 张量维度数
    pad_dim = None
    pad_size = 0
    for i in range(len(pad) // 2):
        left, right = pad[2 * i], pad[2 * i + 1]
        dim = N - 1 - i  # 从后向前映射维度
        if left == 0 and right > 0:
            if pad_dim is not None:
                return None, 0  # 多个维度填充，复杂情况，跳过
            pad_dim = dim
            pad_size = right
        elif left != 0 or right != 0:
            return None, 0  # 复杂填充，跳过
    return pad_dim, pad_size


def get_slice_dim(slice_args, cat_dim):
    """Determine the dimension the slice operates on, relative to cat_dim."""
    if not isinstance(slice_args, tuple) or len(slice_args) < 2:
        return None
    input_rank = len(slice_args)
    for i, sl in enumerate(slice_args):
        if sl is None:
            return None
        if hasattr(sl, 'start') and hasattr(sl, 'stop') and hasattr(sl, 'step'):
            is_slice = isinstance(sl, slice)
            has_bound = sl.start is not None or sl.stop is not None
            has_nontrivial_step = sl.step not in (1, None)
            if is_slice and (has_bound or has_nontrivial_step):
                return i
    return None