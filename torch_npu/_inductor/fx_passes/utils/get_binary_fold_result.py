from typing import Any, Dict, Optional, Union, List
import torch
from torch import fx
from torch.multiprocessing.reductions import StorageWeakRef

from .symbolic_shape_util import (
    has_free_symbols,
    materialize_shape,
    resolve_size_arg,
    shapes_statically_equal,
    statically_known_eq,
    statically_known_geq,
    statically_known_gt,
)


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


def get_node_shape(node: torch.fx.Node, allow_symbolic: bool = False):
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
    if shape is None:
        return None
    # allow_symbolic=False keeps legacy behavior (None if any symbolic dim); True returns the symbolic shape as-is.
    if not allow_symbolic and any(isinstance(s, torch.SymInt) for s in shape):
        return None
    return shape


def get_node_unique_id(node: torch.fx.Node):
    if 'val' in node.meta:
        val = node.meta['val']
        if isinstance(val, torch.Tensor) and (not val.is_sparse or val.layout == torch.strided):
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
        node_meta_shape = get_node_shape(inp, allow_symbolic=True)
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
            if shapes_statically_equal(node_meta_shape, val_meta.shape):
                expand = inp
            else:
                target_shape = list(val_meta.shape)
                if has_free_symbols(target_shape):
                    # Symbolic target shape must be materialized as sym_size refs on
                    # inp's own dims; abandon fold if unmaterializable (e.g. a
                    # broadcast dim originating from the other operand).
                    target_shape = materialize_shape(graph, target_shape, inp)
                    if target_shape is None:
                        return None
                expand = graph.call_function(
                    torch.ops.aten.expand.default,
                    args=(
                        inp,
                        target_shape
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
    # Skip if src is referenced by an in-place op
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
    if not isinstance(dim, int):
        return False
    # start / end may be symbolic (e.g. slicing to a symbolic dim length); normalize to int/SymInt.
    start = resolve_size_arg(start)
    if start is None or not statically_known_eq(start, 0):
        return False
    if end is not None:
        end = resolve_size_arg(end)
        if end is None:
            return False
    src_shape = get_node_shape(src_node, allow_symbolic=True)
    if src_shape is None:
        return False

    node_shape = get_node_shape(node, allow_symbolic=True)
    if node_shape is None:
        return False

    if not shapes_statically_equal(src_shape, node_shape):
        return False

    if dim >= len(src_shape):
        return False

    dim_length = src_shape[dim]

    is_full_slice = (
        end is None
        or statically_known_geq(end, MAX_INT64)
        or statically_known_geq(end, dim_length)
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

    if not isinstance(dim, int):
        return False
    start = resolve_size_arg(start)
    end = resolve_size_arg(end)
    if start is None or end is None:
        return False
    base_shape = get_node_shape(base_node, allow_symbolic=True)
    view_shape = get_node_shape(view_node, allow_symbolic=True)
    if (base_shape is None) or (view_shape is None):
        return False

    if not shapes_statically_equal(base_shape, view_shape):
        return False

    if not statically_known_eq(start, 0):
        return False

    if dim >= len(view_shape):
        return False

    dim_length = view_shape[dim]
    if not statically_known_eq(end, dim_length):
        return False

    node.replace_all_uses_with(view_node)
    propagate_fake_tensor(view_node, node, lambda x: x)
    graph.erase_node(node)
    return True


def get_pad_dim_and_size(pad, input_shape):
    """Extract the padded dim and pad amount from pad args and input shape.

    Args:
        pad: pad list, e.g. [0, 0, 0, max_seq_len].
        input_shape: input tensor shape, e.g. [128, 50, 128].
    Returns:
        pad_dim: padded dim index (0-based).
        pad_size: right-side pad amount.
    """
    N = len(input_shape)  # tensor rank
    pad_dim = None
    pad_size = 0
    for i in range(len(pad) // 2):
        left, right = pad[2 * i], pad[2 * i + 1]
        dim = N - 1 - i  # map back-to-front to dims
        # left/right may be symbolic (e.g. pad to a fixed max_len - s0); use three-valued checks.
        left_zero = statically_known_eq(left, 0)
        if left_zero and statically_known_gt(right, 0):
            if pad_dim is not None:
                return None, 0  # multiple padded dims, complex case -> skip
            pad_dim = dim
            pad_size = right
        elif left_zero and statically_known_eq(right, 0):
            continue  # no padding on this dim
        else:
            return None, 0  # complex or undecidable padding -> skip
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