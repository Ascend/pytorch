from typing import Any, Tuple, Union
import torch
from torch import fx
from torch.fx.operator_schemas import normalize_function, normalize_module
from ...config import log


DTYPE_INSENSITIVE_OPS = {
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.cat.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.gt.Scalar,
    torch.ops.aten.lt.Scalar,
    torch.ops.aten.logical_not.default,
}


CAST_OPS = {
    torch.ops.npu._npu_dtype_cast.default,
    torch.ops.aten.to.dtype,
    torch.ops.prims.convert_element_type.default,
}


def check_cat_op(node: fx.Node):
    is_cat = check_op(node, torch.ops.aten.cat.default)
    if is_cat:
        if len(node.args) == 1:
            return True, 0
        else:
            return True, node.args[1]
    else:
        return False, 0


def check_op(node: fx.Node, target) -> bool:
    return _is_valid_node(node) and node.target == target


def _is_valid_node(node: fx.Node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function"


def is_zero_like(node: Any) -> bool:
    if type(node) in (int, float):
        return node == 0
    elif isinstance(node, fx.Node):
        return node.target in (
            torch.ops.aten.zeros.default,
            torch.ops.aten.zeros_like.default,
        )
    return False


def is_one_like(node: Any) -> bool:
    if type(node) in (int, float):
        return node == 1
    elif isinstance(node, fx.Node):
        return node.target in (
            torch.ops.aten.ones.default,
            torch.ops.aten.ones_like.default,
        )
    return False


def get_input_node(node, idx):
    if node.op == "call_function":
        args_kwargs = normalize_function(node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=False)
    elif node.op == "call_module":
        root = node.graph.owning_module
        args_kwargs = normalize_module(
            root,
            node.target,
            node.args,
            node.kwargs,
            normalize_to_only_use_kwargs=False,
        )
    else:
        args_kwargs = None
    if args_kwargs is not None:
        args, _ = args_kwargs
        if idx < len(args) and idx >= -len(args):
            return args[idx]
    return None


def get_input_kw_node(node, key):
    if node.op == "call_function":
        args_kwargs = normalize_function(node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True)
    elif node.op == "call_module":
        root = node.graph.owning_module
        args_kwargs = normalize_module(root, node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True)
    else:
        args_kwargs = None
    if args_kwargs is not None:
        _, kwargs = args_kwargs
        if key in kwargs:
            return kwargs[key]
    return None


def check_squeeze_op(node: fx.Node) -> bool:
    return (
        check_op(node, torch.ops.aten.squeeze.default)
        or check_op(node, torch.ops.aten.squeeze.dim)
        or check_op(node, torch.ops.aten.squeeze.dims)
    )


def check_unsqueeze_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.unsqueeze.default)


def check_where_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.where.self)


def check_mul_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.mul.Tensor)


def check_view(node):
    if (not check_op(node, torch.ops.aten._unsafe_view.default)) and (not check_op(node, torch.ops.aten.view.default)) and (not check_op(node, torch.ops.aten.reshape.default)):
        return False
    return True


def check_act_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[Tuple[Union[fx.Node, None], bool], None]]:
    if not _is_valid_node(node):
        return False, None
    if node.target == torch.ops.aten.silu.default:
        return True, "silu"
    if node.target == torch.ops.aten.gelu.default:
        return True, "gelu"
    if node.target == torch.ops.aten.relu.default:
        return True, "relu"
    if node.target == torch.ops.aten.sigmoid.default:
        return True, "sigmoid"
    return False, None


def check_add_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.add.Tensor)


def check_sub_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.sub.Tensor)


def check_div_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.div.Tensor)


def check_embedding_op(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    
    embedding_targets = {
        torch.nn.functional.embedding, 
        torch.ops.aten.embedding.default, 
        torch.embedding, 
    }
    return node.target in embedding_targets


def check_op_by_targets(node: fx.Node, targets) -> bool:
    for target in targets:
        result = check_op(node, target)
        if not result:
            continue
        return True
    return False


def try_match(lhs, rhs, identity_fn):
    if identity_fn(lhs):
        return True, rhs
    if identity_fn(rhs):
        return True, lhs
    return False, None


def normalize_dtype(dt):
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        try:
            return getattr(torch, dt)
        except Exception as e:
            log.debug(f"normalize_dtype catch exception {e}")
            return None
    return None


def is_cast_node(node):
    return (
        node.op == "call_function"
        and node.target in CAST_OPS
    )


def get_cast_dtype(node):
    if node.target in CAST_OPS:
        return node.args[1] if len(node.args) > 1 else None
    return None


def get_node_dtype(node):
    val = node.meta.get('val', None)
    if val is not None and hasattr(val, 'dtype'):
        return val.dtype
    tmeta = node.meta.get('tensor_meta', None)
    if tmeta is not None and hasattr(tmeta, "dtype"):
        return tmeta.dtype
    return node.meta.get("dtype", None)


def check_support_op(user):
    return check_add_op(user) or check_sub_op(user) or check_mul_op(user) or check_div_op(user)


def _get_tensor_meta(node):
    return node.meta.get("tensor_meta", None)