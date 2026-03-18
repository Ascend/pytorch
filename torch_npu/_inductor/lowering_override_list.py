import torch
from .config import inductor_indirect_memory_mode

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu


# in lowering.py, we will remove the following op's default lowering function, 
# and register its new override-lowering-function.
LOWERING_OVERRIDE_OP = [
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
    torch.ops.higher_order.flex_attention,
    torch.ops.higher_order.flex_attention_backward,
]

INDIRECT_MEM_OVERRIDE_LIST = [
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

if inductor_indirect_memory_mode == "simt_template":
    LOWERING_OVERRIDE_OP += INDIRECT_MEM_OVERRIDE_LIST
