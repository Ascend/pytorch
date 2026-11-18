import torch

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu


# in lowering.py, we will remove the following op's default lowering function,
# and register its new override-lowering-function.
LOWERING_OVERRIDE_OP = [
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
    aten.fractional_max_pool3d,
    aten.pow,
    prims.convert_element_type,
    aten.mm,
    aten.bmm,
    aten.addmm,
    torch.ops.higher_order.flex_attention,
    torch.ops.higher_order.flex_attention_backward,
]
