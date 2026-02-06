import torch
import torch.fx
import torch.utils._pytree as pytree

aten = torch.ops.aten
prims = torch.ops.prims

DVM_OP_REGISTRY = {}

DVM_SUPPORT_TYPE = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.int32,
    torch.bool,
]


DVM_SUPPORT_FLOAT_TYPE = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
]

DVM_DTYPE_MAP = {
    torch.bfloat16: "dvm.bfloat16",
    torch.float16: "dvm.float16",
    torch.float32: "dvm.float32",
    torch.int32: "dvm.int32",
    torch.int64: "dvm.int64",
    torch.bool: "dvm.bool_",
}


def to_dvm_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        if dtype not in DVM_DTYPE_MAP:
            raise NotImplementedError(f"Unsupported dtype for DVM: {dtype}")
        return DVM_DTYPE_MAP[dtype]
    return dtype


def _check_dtype(inputs, supported_dtypes):
    for inp in inputs:
        if not isinstance(inp, torch.fx.Node):
            continue
        if "val" not in inp.meta:
            continue

        for meta in pytree.tree_leaves(inp.meta["val"]):
            if not isinstance(meta, torch._subclasses.FakeTensor):
                continue
            if meta.dtype not in supported_dtypes:
                return False
            if meta.is_cpu:
                return False
    return True


def common_rule(node: torch.fx.Node):
    return _check_dtype(
        pytree.arg_tree_leaves(
            *node.args, **node.kwargs), DVM_SUPPORT_FLOAT_TYPE
    ) and _check_dtype([node], supported_dtypes=DVM_SUPPORT_TYPE)


def full_rule(node: torch.fx.Node):
    return _check_dtype([node], supported_dtypes=DVM_SUPPORT_TYPE)


def cast_rule(node: torch.fx.Node):
    return _check_dtype(node.args, supported_dtypes=DVM_SUPPORT_TYPE) and _check_dtype(
        [node], supported_dtypes=DVM_SUPPORT_TYPE
    )


def where_rule(node: torch.fx.Node):
    return _check_dtype(node.args[1:], DVM_SUPPORT_FLOAT_TYPE) and _check_dtype(
        [node], supported_dtypes=DVM_SUPPORT_TYPE
    )


def _is_last2_transpose_tensor(t: torch._subclasses.FakeTensor) -> bool:
    if t.dim() < 2:
        return False

    if not (t.stride(-2) == 1 and t.stride(-1) == t.size(-2)):
        return False

    batch = t.size(-1) * t.size(-2)
    for d in range(t.dim() - 3, -1, -1):
        if t.stride(d) != batch:
            return False
        batch *= t.size(d)

    return True


def mm_rule(node: torch.fx.Node):
    UINT16_MAX = (1 << 16) - 1
    UINT8_MAX = (1 << 8) - 1
    MAX_INNER = UINT16_MAX - UINT8_MAX

    def inner_axis_length(t: torch._subclasses.FakeTensor):
        if _is_last2_transpose_tensor(t):
            return t.mT.size(-1)
        return t.size(-1)

    def check(input_node):
        t = input_node.meta["val"]
        inner_axis = inner_axis_length(t)
        if isinstance(inner_axis, torch.SymInt):
            return False
        if t.dim() > 4 or t.dim() < 2 or inner_axis > MAX_INNER:
            return False
        return True

    if node.target in [aten.mm.default, aten.bmm.default]:
        lhs = node.args[0]
        rhs = node.args[1]
    elif node.target is aten.addmm.default:
        lhs = node.args[1]
        rhs = node.args[2]
    else:
        return False
    if node.meta["val"].dtype not in (torch.float16, torch.bfloat16):
        return False

    return check(lhs) and check(rhs)


def register_dvm_op(*ops, rule=common_rule):
    def decorator(func):
        for op in ops:
            DVM_OP_REGISTRY[op] = (func, rule)
        return func

    return decorator


def format_shape(shape):
    if isinstance(shape, (int, torch.SymInt)):
        shape = [shape]
    return "[" + ", ".join(map(str, shape)) + "]"


@register_dvm_op(aten.add.Tensor, aten.add.Scalar)
def add(x, y):
    return f"k.add({x}, {y})"


@register_dvm_op(aten.sub.Tensor, aten.sub.Scalar)
def sub(x, y):
    return f"k.sub({x}, {y})"


@register_dvm_op(aten.mul.Tensor, aten.mul.Scalar)
def mul(x, y):
    return f"k.mul({x}, {y})"


@register_dvm_op(aten.div.Tensor, aten.div.Scalar)
def div(x, y):
    return f"k.div({x}, {y})"


@register_dvm_op(aten.pow.Tensor_Tensor, aten.pow.Tensor_Scalar, aten.pow.Scalar)
def pow_op(x, y):
    return f"k.pow({x}, {y})"


@register_dvm_op(aten.lt.Tensor, aten.lt.Scalar)
def less(x, y):
    return f"k.less({x}, {y})"


@register_dvm_op(aten.le.Tensor, aten.le.Scalar)
def less_equal(x, y):
    return f"k.less_equal({x}, {y})"


@register_dvm_op(aten.gt.Tensor, aten.gt.Scalar)
def greater(x, y):
    return f"k.greater({x}, {y})"


@register_dvm_op(aten.ge.Tensor, aten.ge.Scalar)
def greater_equal(x, y):
    return f"k.greater_equal({x}, {y})"


@register_dvm_op(aten.maximum.default)
def maximum(x, y):
    return f"k.maximum({x}, {y})"


@register_dvm_op(aten.minimum.default)
def minimum(x, y):
    return f"k.minimum({x}, {y})"


@register_dvm_op(aten.clamp_min.default)
def clamp_min(x, min_value):
    return maximum(x, min_value)


@register_dvm_op(aten.clamp_max.default)
def clamp_max(x, max_value):
    return minimum(x, max_value)


@register_dvm_op(aten.logical_and.default)
def logical_and(x, y):
    return f"k.logical_and({x}, {y})"


@register_dvm_op(aten.logical_or.default)
def logical_or(x, y):
    return f"k.logical_or({x}, {y})"


@register_dvm_op(aten.eq.Tensor, aten.eq.Scalar)
def equal(x, y):
    return f"k.equal({x}, {y})"


@register_dvm_op(aten.ne.Tensor, aten.ne.Scalar)
def not_equal(x, y):
    return f"k.not_equal({x}, {y})"


@register_dvm_op(aten.sqrt.default)
def sqrt(x):
    return f"k.sqrt({x})"


@register_dvm_op(aten.rsqrt.default)
def rsqrt(x):
    return div(1, sqrt(x))


@register_dvm_op(aten.abs.default)
def abs_op(x):
    return f"k.abs({x})"


@register_dvm_op(aten.log.default)
def log(x):
    return f"k.log({x})"


@register_dvm_op(aten.exp.default)
def exp(x):
    return f"k.exp({x})"


@register_dvm_op(aten.reciprocal.default)
def reciprocal(x):
    return f"k.reciprocal({x})"


@register_dvm_op(aten.isfinite.default)
def is_finite(x):
    return f"k.is_finite({x})"


@register_dvm_op(aten.logical_not.default)
def logical_not(x):
    return f"k.logical_not({x})"


@register_dvm_op(aten.round.default, aten.round.decimals)
def round_op(x):
    return f"k.round({x})"


@register_dvm_op(aten.floor.default)
def floor(x):
    return f"k.floor({x})"


@register_dvm_op(aten.ceil.default)
def ceil(x):
    return f"k.ceil({x})"


@register_dvm_op(aten.trunc.default)
def trunc(x):
    return f"k.trunc({x})"


@register_dvm_op(
    prims.convert_element_type.default,
    torch.ops.npu.npu_dtype_cast.default,
    torch.ops.npu.npu_dtype_cast_backward.default,
    torch.ops.npu._npu_dtype_cast.default,
    torch.ops.npu._npu_dtype_cast_backward.default,
    rule=cast_rule,
)
def cast(x, dtype):
    dtype = to_dvm_dtype(dtype)
    return f"k.cast({x}, {dtype})"


@register_dvm_op(aten.expand.default)
def broadcast(x, shape):
    shape = format_shape(shape)
    return f"k.broadcast({x}, {shape})"


@register_dvm_op(aten.where.default, aten.where.self, rule=where_rule)
def select(x, y, z):
    return f"k.select({x}, {y}, {z})"


@register_dvm_op(aten.sum.dim_IntList, aten.sum.default)
def reduce_sum(x, dim=None, keepdim=False, dtype=None):
    if dim is None:
        dim = []
    dim = format_shape(dim)
    return f"k.sum({x}, {dim}, {keepdim})"


@register_dvm_op(aten.amax.default)
def reduce_max(x, dim=None, keepdim=False):
    if dim is None:
        dim = []
    dim = format_shape(dim)
    return f"k.max({x}, {dim}, {keepdim})"


@register_dvm_op(aten.amin.default)
def reduce_min(x, dim=None, keepdim=False):
    if dim is None:
        dim = []
    dim = format_shape(dim)
    return f"k.min({x}, {dim}, {keepdim})"


@register_dvm_op(aten.view.default, aten.reshape.default)
def reshape(x, shape):
    shape = format_shape(shape)
    return f"k.reshape({x}, {shape})"


@register_dvm_op(aten.neg.default)
def neg(x):
    return mul(x, -1)


@register_dvm_op(aten.relu.default)
def relu(x):
    return maximum(x, 0)


def copy(x):
    return f"k.copy({x})"


@register_dvm_op(aten.clone.default)
def clone(x, memory_format=None):
    return copy(x)


@register_dvm_op(aten.full.default, rule=full_rule)
def full(
    size,
    fill_value,
    **kwargs,
):
    size = format_shape(size)
    dtype = to_dvm_dtype(kwargs.get("dtype"))
    return f"k.full({fill_value}, {size}, {dtype})"


@register_dvm_op(aten.mm.default, aten.bmm.default, rule=mm_rule)
def matmul(x, y, trans_a, trans_b):
    return f"k.matmul({x}, {y}, {trans_a}, {trans_b})"


def matmul_bias(bias, x, y, trans_a, trans_b, beta=1, alpha=1):
    return f"k.matmul({x}, {y}, {trans_a}, {trans_b},{bias})"


@register_dvm_op(aten.addmm.default, rule=mm_rule)
def addmm(z, x, y, trans_a, trans_b, use_bias, beta=1, alpha=1):
    if use_bias:
        return matmul_bias(z, x, y, trans_a, trans_b)
    if beta != 1:
        z = mul(z, beta)
    mm = matmul(x, y, trans_a, trans_b)
    if alpha != 1:
        mm = mul(mm, alpha)
    return add(mm, z)


def load(shape, dtype):
    dtype = to_dvm_dtype(dtype)
    return f"k.load({shape}, {dtype})"


def view_load(shape, stride, offset, dtype):
    dtype = to_dvm_dtype(dtype)
    return f"k.view_load({shape}, {stride}, {offset}, {dtype})"


def store(x, dtype=None):
    if dtype is None:
        return f"k.store({x})"
    dtype = to_dvm_dtype(dtype)
    return f"k.store({x}, {dtype})"
