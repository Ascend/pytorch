import torch._ops
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._inductor.decomposition import register_decomposition
from torch._prims_common.wrappers import out_wrapper
from .config import get_soc_version, Ascend910_9391
from .lowering import _init_set

aten = torch.ops.aten

DECOMPOSITION_OVERLOAD_OP = [
    aten.nll_loss_forward,
    # aten.gelu_backward,
    # aten.gelu,
    aten.nll_loss_backward,
    aten._log_softmax_backward_data,
    aten.embedding_dense_backward,
    aten.addmm,
    aten.gelu,
    aten.native_layer_norm,
    aten.native_dropout,
    aten.native_dropout_backward
]

if get_soc_version() >= Ascend910_9391:
    DECOMPOSITION_OVERLOAD_OP.append(aten.max_pool2d_with_indices)


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

    @register_decomposition(aten.native_dropout)
    @out_wrapper("out0", "out1")
    def native_dropout(tensor_input, p, train):
        if torch._inductor.config.fallback_random:
            if train and p != 0:
                return torch.ops.npu._npu_dropout(tensor_input, p)
            return (tensor_input, torch.ones_like(tensor_input, dtype=torch.bool))
        else:
            from torch._decomp.decompositions import native_dropout
            return native_dropout(tensor_input, p, train)

    @register_decomposition(aten.native_dropout_backward)
    @out_wrapper()
    def native_dropout_backward(grad_output, mask, scale):
        if torch._inductor.config.fallback_random:
            p = 1 if scale == 0 else (1 - 1 / scale)
            r = torch.ops.npu.npu_dropout_backward(grad_output, mask, p)
            return r
        else:
            from torch._decomp.decompositions import native_dropout_backward
            return native_dropout_backward(grad_output, mask, scale)
