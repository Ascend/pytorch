import torch._ops
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._inductor.decomposition import register_decomposition
from torch._prims_common.wrappers import out_wrapper
from .config import get_soc_version, is_ascend950
from .lowering import _add_overload

aten = torch.ops.aten

DECOMPOSITION_OVERLOAD_OP = [
    aten.nll_loss_forward,
    aten.nll_loss_backward,
    aten._log_softmax_backward_data,
    # aten.embedding_dense_backward,
    aten.addmm,
    aten.gelu,
    aten.native_layer_norm,
]

if is_ascend950:
    DECOMPOSITION_OVERLOAD_OP.append(aten.max_pool2d_with_indices)


def _register_npu_inductor_decompositons():
    overload_op_set = set()
    _add_overload(DECOMPOSITION_OVERLOAD_OP, overload_op_set)

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