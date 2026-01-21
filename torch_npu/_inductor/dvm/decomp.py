import math
import torch
from torch._inductor import decomposition as inductor_decomp
from torch._decomp import remove_decompositions
aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized

decomps_to_exclude_npu = [
    aten.gelu.default,
    aten.gelu_backward.default,
    aten.nll_loss_forward,
    aten.nll_loss_backward,
    aten.linalg_vector_norm,
    aten._log_softmax,
    aten._log_softmax_backward_data,
    aten.embedding_dense_backward,
    aten._native_batch_norm_legit,
    aten._native_batch_norm_legit_functional,
    aten._native_batch_norm_legit_no_training,
    aten._batch_norm_with_update,
    aten._batch_norm_with_update_functional,
    aten._batch_norm_no_update,
    aten.native_batch_norm,
    aten.batch_norm_backward,
    aten.native_group_norm,
    aten.native_layer_norm,
    aten.native_layer_norm_backward,
    aten.convolution_backward,
    aten._softmax,
    aten._softmax_backward_data,
    torch.ops.npu.npu_rotary_mul,
    torch.ops.npu.npu_rotary_mul_backward,
]


def tanh(a):
    """
    tanh(a) = (exp(a) - exp(-a)) / (exp(a) + exp(-a))
    """
    orig_dtype = a.dtype
    if orig_dtype != torch.float32:
        a = a.to(torch.float32)
    ea = torch.exp(a)
    e_minus_a = torch.exp(-a)
    out = (ea - e_minus_a) / (ea + e_minus_a)
    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


def gelu(a: torch.Tensor, approximate: str = "none"):
    """
      y = -sqrt(8/pi) * (x + 0.044715 * x^3)
      out = x / (1 + exp(y))
    """
    orig_dtype = a.dtype
    if orig_dtype != torch.float32:
        a = a.to(torch.float32)

    M_SQRT2 = math.sqrt(2)   
    M_2_SQRTPI = 2.0 / math.sqrt(math.pi)
    kBeta = M_SQRT2 * M_2_SQRTPI
    kKappa = 0.044715

    a_cube = a * a * a
    inner = a + kKappa * a_cube
    y = -kBeta * inner
    out = a / (1.0 + torch.exp(y))

    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


def gelu_backward(grad, self, approximate: str = "none"):
    orig_dtype = grad.dtype
    if orig_dtype != torch.float32:
        grad = grad.to(torch.float32)
        self = self.to(torch.float32)
    M_SQRT2 = math.sqrt(2)
    M_2_SQRTPI = 2.0 / math.sqrt(math.pi)
    kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
    kKappa = 0.044715
    x_sq = self * self
    x_cube = x_sq * self
    inner = kBeta * (self + kKappa * x_cube)
    tanh_inner = torch.tanh(inner)

    left = 0.5 * self
    right = 1.0 + tanh_inner

    left_derivative = 0.5 * right

    tanh_derivative = (tanh_inner * tanh_inner) * -1.0 + 1.0
    inner_derivative = kBeta * (1.0 + 3.0 * kKappa * x_sq)
    right_derivative = left * tanh_derivative * inner_derivative
    out = grad * (left_derivative + right_derivative)

    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


def sigmoid(a: torch.Tensor) -> torch.Tensor:
    orig_dtype = a.dtype
    if orig_dtype != torch.float32:
        a = a.to(torch.float32)
    out = 1 / (1.0 + torch.exp(-a))
    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


def matmul_backward_decomposition(grad, self, other, mask):
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
        grad_self = grad.unsqueeze(1).mm(
            other.unsqueeze(0)) if mask[0] else grad_self
        grad_other = self.transpose(-1, -2).mm(grad.unsqueeze(1)
                                               ).squeeze_(1) if mask[1] else grad_other
        return grad_self, grad_other

    def matmul_backward_1d_2d():
        nonlocal grad_self, grad_other
        grad_self = grad.unsqueeze(0).mm(
            other.transpose(-1, -2)).squeeze_(0) if mask[0] else grad_self
        grad_other = self.unsqueeze(1).mm(
            grad.unsqueeze(0)) if mask[1] else grad_other
        return grad_self, grad_other

    def matmul_backward_nd_lt3d():
        nonlocal grad_self, grad_other
        view_size = 1 if dim_other == 1 else size_grad[-1]
        unfolded_grad = (grad.unsqueeze(-1) if dim_other ==
                         1 else grad).contiguous().view(-1, view_size)
        if mask[0]:
            unfolded_other = other.unsqueeze(
                0) if dim_other == 1 else other.transpose(-1, -2)
            grad_self = unfolded_grad.mm(unfolded_other).view(size_self)

        if mask[1]:
            # create a 2D-matrix from self
            unfolded_self = self.contiguous().view(-1, size_self[-1])
            grad_other = unfolded_self.transpose(-1, -
                                                 2).mm(unfolded_grad).view(size_other)
        return grad_self, grad_other

    def matmul_backward_lt3d_nd():
        nonlocal grad_self, grad_other
        view_size = 1 if dim_self == 1 else size_grad[-2]
        unfolded_grad_t = grad.view(-1, view_size) if dim_self == 1 else \
            grad.transpose(-1, -2).contiguous().view(-1, view_size)
        if mask[0]:
            # create a 2D-matrix from other
            unfolded_other_t = \
                other.transpose(-1, -2).contiguous().view(-1,
                                                          size_other[-2]).transpose(-1, -2)
            grad_self = unfolded_other_t.mm(
                unfolded_grad_t).transpose(-1, -2).view(size_self)

        if mask[1]:
            size_other_t = list(size_other[:-2])
            size_other_t.extend(
                [size_other[dim_other - 1], size_other[dim_other - 2]])
            unfolded_self = self.unsqueeze(0) if dim_self == 1 else self
            grad_other = unfolded_grad_t.mm(unfolded_self).view(
                size_other_t).transpose(-1, -2)
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
        grad_self = torch.matmul(
            grad, other.transpose(-1, -2)) if mask[0] else grad_self
        grad_other = torch.matmul(
            self.transpose(-1, -2), grad) if mask[1] else grad_other

    return grad_self, grad_other


def patch_decomp():
    remove_decompositions(inductor_decomp.decompositions,
                          decomps_to_exclude_npu)
    inductor_decomp.register_decomposition([aten.sigmoid.default])(sigmoid)
    inductor_decomp.register_decomposition(
        [aten.gelu_backward.default])(gelu_backward)
    inductor_decomp.register_decomposition([aten.gelu.default])(gelu)
    inductor_decomp.register_decomposition([aten.tanh.default])(tanh)
    inductor_decomp.register_decomposition(
        [aten.matmul_backward.default])(matmul_backward_decomposition)
