import torch._ops
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._inductor.decomposition import register_decomposition

from .lowering import _init_set

aten = torch.ops.aten

DECOMPOSITION_OVERLOAD_OP = [
    aten._log_softmax,
    aten.nll_loss_forward,
    # aten.gelu_backward,
    # aten.gelu,
    aten.nll_loss_backward,
    aten._log_softmax_backward_data,
    aten.embedding_dense_backward,
    aten.addmm,
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
        tensor = torch.ones_like(x) - torch.exp(x)
        return tensor
