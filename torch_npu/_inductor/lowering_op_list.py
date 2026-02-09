import torch
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from .config import inductor_indirect_memory_mode

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu

GENERATE_LIST = [
    prims.device_put,
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
    npu.npu_dtype_cast,
    npu._npu_dtype_cast,
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
    aten.copy_,
    aten.reciprocal,
    aten._assert_scalar,
    triton_kernel_wrapper_mutation,
    aten.native_layer_norm,
    aten.mm,
    aten.bmm,
    aten.addmm,
    npu.npu_grouped_matmul,
]

GENERATE_LIST2 = [
    "foreach"
]


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

    aten.var_mean,
    aten.var,

    aten.cat,
    aten.mm,
    aten.bmm,
    aten.addmm,
]

INDIRECT_MEM_GENERATE_LIST = [
    aten.embedding,
    aten.gather,
    aten.index_put_,
    aten.index_put,
    aten._unsafe_index_put,
    aten.scatter,
    aten.scatter_,
    aten.scatter_reduce,
    aten.scatter_reduce_,
    aten.index,
    aten._unsafe_index
]

INDIRECT_MEM_OVERLOAD_LIST = [
    aten.embedding,
    aten.gather,
    aten.index,
    aten._unsafe_index,
    aten.index_put_,
    aten.index_put,
    aten._unsafe_index_put,
    aten.scatter,
    aten.scatter_,
    aten.scatter_reduce,
    aten.scatter_reduce_,
]

if inductor_indirect_memory_mode:
    GENERATE_LIST += INDIRECT_MEM_GENERATE_LIST
    if inductor_indirect_memory_mode == "simt_template":
        LOWERING_OVERLOAD_OP += INDIRECT_MEM_OVERLOAD_LIST
