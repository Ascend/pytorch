import torch._ops
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath	
from torch._inductor.decomposition import register_decomposition

from .lowering import _init_set

aten = torch.ops.aten

DECOMPOSITION_OVERLOAD_OP = [
    aten.gelu
]


def _register_npu_inductor_decompositons():
    overload_op_set = set()
    _init_set(DECOMPOSITION_OVERLOAD_OP, overload_op_set)

    for op in overload_op_set:
        if (op in decompositions):
            del decompositions[op]

    @register_decomposition([aten.expm1])
    def expm1(x):
        tensor = torch.exp(x) - torch.ones_like(x)
        return tensor

    @register_decomposition([aten.erfc])
    def erfc(x):
        tensor = torch.ones_like(x) - torch.erf(x)
        return tensor
