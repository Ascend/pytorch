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

FP32_MIN_V2 = -8.8
FP32_MAX_V2 = 8.8
DOUBLE_X = 2.0


def tanh(a):
    """
    y = (exp(2x) - 1) / (exp(2x) + 1)
    with x clipped to [-8.8, 8.8] in float32 before multiply-by-2.
    """
    orig_dtype = a.dtype
    if orig_dtype != torch.float32:
        a = a.to(torch.float32)
    x = torch.clamp(a, min=FP32_MIN_V2, max=FP32_MAX_V2)
    x2 = x * DOUBLE_X
    e2x = torch.exp(x2)
    out = (e2x - 1.0) / (e2x + 1.0)

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
    out = 1 / (1.0 + torch.exp(torch.neg(a)))
    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


def patch_decomp():
    remove_decompositions(inductor_decomp.decompositions, decomps_to_exclude_npu)
    inductor_decomp.register_decomposition([aten.sigmoid.default])(sigmoid)
    inductor_decomp.register_decomposition([aten.gelu_backward.default])(gelu_backward)
    inductor_decomp.register_decomposition([aten.gelu.default])(gelu)
    inductor_decomp.register_decomposition([aten.tanh.default])(tanh)
