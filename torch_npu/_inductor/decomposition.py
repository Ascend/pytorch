import torch._ops
from torch._inductor.decomposition import pw_cast_for_opmath, register_decomposition

aten = torch.ops.aten


def _register_npu_inductor_decompositons():
    @register_decomposition([aten.scatter.src])
    @pw_cast_for_opmath
    def scatter_src(self, input_tensor, dim, index_tensor, source_tensor):
        (XNUMEL, YS) = input_tensor.shape
        index_rblock = torch.arange(YS).npu().reshape((1, YS)).repeat((XNUMEL, 1))

        index_tensor_brd = index_tensor.to(torch.int32).broadcast_to(XNUMEL, YS)
        source_tensor_brd = source_tensor.broadcast_to(XNUMEL, YS).to(torch.float32)
        scatter1 = torch.where(index_rblock == index_tensor_brd, 1.0, 0.0) * source_tensor_brd
        return scatter1

    @register_decomposition([aten.expm1])
    def expm1(x):
        tensor = torch.exp(x) - torch.ones_like(x)
        return tensor

    @register_decomposition([aten.erfc])
    def erfc(x):
        tensor = torch.ones_like(x) - torch.erf(x)
        return tensor
