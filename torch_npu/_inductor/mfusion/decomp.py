import torch
from torch._decomp import remove_decompositions
from torch._inductor import decomposition as inductor_decomp


aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized

decomps_to_exclude_npu = [
    aten._batch_norm_no_update,
    aten._batch_norm_with_update,
    aten._batch_norm_with_update_functional,
    aten._log_softmax,
    aten._log_softmax_backward_data,
    aten._softmax,
    aten._softmax_backward_data,
    aten.batch_norm_backward,
    aten.convolution_backward,
    aten.embedding,
    aten.embedding_backward,
    aten.embedding_dense_backward,
    aten.gelu.default,
    aten.gelu_backward.default,
    aten.grid_sampler_2d,
    aten.grid_sampler_2d_backward,
    aten.linalg_vector_norm,
    aten.max_pool2d_with_indices,
    aten.max_pool2d_with_indices_backward,
    aten.native_batch_norm,
    aten.native_group_norm,
    aten.native_layer_norm,
    aten.nll_loss2d_backward,
    aten.nll_loss2d_forward,
    aten.nll_loss_backward,
    aten.nll_loss_forward,
    aten.reflection_pad2d,
    aten.reflection_pad2d_backward,
    aten.rms_norm,
    aten.slice.Tensor,
    aten.triu,
    aten.upsample_bilinear2d,
    aten.upsample_bilinear2d_backward,
    aten.upsample_nearest1d,
    aten.upsample_nearest1d_backward,
    aten.upsample_nearest2d,
    aten.upsample_nearest2d_backward,
    aten.upsample_nearest3d,
    aten.upsample_nearest3d_backward,
    torch.ops.npu.npu_rotary_mul,
    torch.ops.npu.npu_rotary_mul_backward,
]


def matmul_backward(grad, self, other, mask):
    dim_self = self.dim()
    dim_other = other.dim()

    size_grad = grad.size()
    size_self = self.size()
    size_other = other.size()
    grad_self = None
    grad_other = None

    def matmul_backward_1d_1d():
        nonlocal grad_self, grad_other
        grad_self = other.mul(grad) if mask[0] else grad_self
        grad_other = self.mul(grad) if mask[1] else grad_other
        return grad_self, grad_other

    def matmul_backward_2d_1d():
        nonlocal grad_self, grad_other
        grad_self = grad.unsqueeze(1).mm(other.unsqueeze(0)) if mask[0] else grad_self
        grad_other = (
            self.transpose(-1, -2).mm(grad.unsqueeze(1)).squeeze_(1)
            if mask[1]
            else grad_other
        )
        return grad_self, grad_other

    def matmul_backward_1d_2d():
        nonlocal grad_self, grad_other
        grad_self = (
            grad.unsqueeze(0).mm(other.transpose(-1, -2)).squeeze_(0)
            if mask[0]
            else grad_self
        )
        grad_other = self.unsqueeze(1).mm(grad.unsqueeze(0)) if mask[1] else grad_other
        return grad_self, grad_other

    def matmul_backward_nd_lt3d():
        nonlocal grad_self, grad_other
        view_size = 1 if dim_other == 1 else size_grad[-1]
        unfolded_grad = (
            (grad.unsqueeze(-1) if dim_other == 1 else grad)
            .contiguous()
            .view(-1, view_size)
        )
        if mask[0]:
            unfolded_other = (
                other.unsqueeze(0) if dim_other == 1 else other.transpose(-1, -2)
            )
            grad_self = unfolded_grad.mm(unfolded_other).view(size_self)

        if mask[1]:
            # create a 2D-matrix from self
            unfolded_self = self.contiguous().view(-1, size_self[-1])
            grad_other = (
                unfolded_self.transpose(-1, -2).mm(unfolded_grad).view(size_other)
            )
        return grad_self, grad_other

    def matmul_backward_lt3d_nd():
        nonlocal grad_self, grad_other
        view_size = 1 if dim_self == 1 else size_grad[-2]
        unfolded_grad_t = (
            grad.view(-1, view_size)
            if dim_self == 1
            else grad.transpose(-1, -2).contiguous().view(-1, view_size)
        )
        if mask[0]:
            # create a 2D-matrix from other
            unfolded_other_t = (
                other.transpose(-1, -2)
                .contiguous()
                .view(-1, size_other[-2])
                .transpose(-1, -2)
            )
            grad_self = (
                unfolded_other_t.mm(unfolded_grad_t).transpose(-1, -2).view(size_self)
            )

        if mask[1]:
            size_other_t = list(size_other[:-2])
            size_other_t.extend([size_other[dim_other - 1], size_other[dim_other - 2]])
            unfolded_self = self.unsqueeze(0) if dim_self == 1 else self
            grad_other = (
                unfolded_grad_t.mm(unfolded_self).view(size_other_t).transpose(-1, -2)
            )
        return grad_self, grad_other

    if dim_self == 1 and dim_other == 1:
        grad_self, grad_other = matmul_backward_1d_1d()
    elif dim_self == 2 and dim_other == 1:
        grad_self, grad_other = matmul_backward_2d_1d()
    elif dim_self == 1 and dim_other == 2:
        grad_self, grad_other = matmul_backward_1d_2d()
    elif dim_self >= 3 and (dim_other == 1 or dim_other == 2):
        # create a 2D-matrix from grad
        grad_self, grad_other = matmul_backward_nd_lt3d()
    elif (dim_self == 1 or dim_self == 2) and dim_other >= 3:
        # create a 2D-matrix from grad
        grad_self, grad_other = matmul_backward_lt3d_nd()
    else:
        grad_self = (
            torch.matmul(grad, other.transpose(-1, -2)) if mask[0] else grad_self
        )
        grad_other = (
            torch.matmul(self.transpose(-1, -2), grad) if mask[1] else grad_other
        )

    return grad_self, grad_other


def fma(a, b, c):
    return a * b + c


_mfusion_inductor_decomp_patched = False


def _register_inductor_decomposition_safe(overloads, fn):
    try:
        inductor_decomp.register_decomposition(overloads)(fn)
    except (RuntimeError, ValueError) as e:
        msg = str(e).lower()
        if any(
            s in msg
            for s in (
                "duplicate",
                "already",
                "exists",
                "re-register",
                "re_register",
            )
        ):
            return
        raise


def patch_decomp():
    """Patch Inductor decomposition table for NPU MFusion (idempotent).

    May be invoked from MFusionPatch.enable() more than once in the same
    process (e.g. multiple unit tests). PyTorch raises on duplicate
    register_decomposition for the same overload.
    """
    global _mfusion_inductor_decomp_patched
    if _mfusion_inductor_decomp_patched:
        return
    remove_decompositions(inductor_decomp.decompositions, decomps_to_exclude_npu)
    _register_inductor_decomposition_safe(
        [aten.matmul_backward.default], matmul_backward
    )
    _register_inductor_decomposition_safe([prims.fma.default], fma)
    _mfusion_inductor_decomp_patched = True
