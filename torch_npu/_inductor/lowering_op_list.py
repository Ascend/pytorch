import torch
from torch_npu import npu_dtype_cast

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims

GENERATE_LIST = [
    prims.iota,
    aten.full,
    aten.mul,
    aten.add,
    aten.sub,
    aten.div,
    aten.exp,
    aten.maximum,
    aten.sum,
    aten.select,
    aten.unsqueeze,
    aten.repeat,
    aten.clone,
    aten.reshape,
    aten.where,
    aten.lt,
    aten.minimum,
    aten.gt,
    aten.le,
    aten.ceil,
    aten.floor,
    aten.rsqrt,
    aten.abs,
    aten.log,
    aten.bitwise_xor,
    aten.amax,
    # backward
    prims.convert_element_type,
    aten.min,
    aten.max,
    aten.erf,
    aten.argmax,
    aten.argmin,
    aten.clamp_min,
    aten.slice,
    aten.neg,
    aten.cat,
    aten.arange,
    aten.expand,
    aten.eq,
    aten.where,
    aten.scalar_tensor,
    aten.ge,
    aten.permute,
    aten.sqrt,
    aten.relu,
    aten.clamp,
    aten.clamp_max,
    aten.mean,
    npu_dtype_cast,
    aten.select_scatter,
    aten.slice_scatter,
    prims.broadcast_in_dim,
    prims.maximum,
    aten.ne,
    aten.sigmoid,
    aten.sign,
    aten.logical_and,
    aten.logical_or,
    aten.logical_not,
    aten.pow,
    aten.gelu,
    aten.tanh,
    aten.isnan,
    aten.bitwise_and,
    aten.squeeze,
    aten.copy,
    aten.reciprocal
]

GENERATE_LIST2 = [
    "foreach"
]

FALLBACK_LIST = []

# Delete these op in lowering list and then update lowering list with new lowering,
# otherwise, it will not use npu overload lowering.
LOWERING_OVERLOAD_OP = [
    aten.cumsum,
    aten.mean,
    aten.max,
    aten.min,
    aten.amin,
    aten.amax,
    aten.argmax,
    aten.argmin,
    aten.sum,

    aten.var_mean,
    aten.var,

    aten.embedding,
    aten.split,
    aten.split_with_sizes,
    aten.nll_loss_forward,
    aten.gather,
    aten.cat,
]
