import functools
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch._inductor import decomposition as inductor_decomp
from torch._C import DispatchKey
from torch import Tensor

from torch._decomp import (
    remove_decompositions,
)

from .. import config as anir_config

aten = torch.ops.aten
npu = torch.ops.npu

remove_decompositions(inductor_decomp.decompositions, anir_config.decomps_to_exclude_npu)

# Batch_norm_decomposition function registered to fix dynamic shape dynamo tracing issue.
@aten.batch_norm.default.py_impl(DispatchKey.Autograd)
@aten.batch_norm.default.py_impl(DispatchKey.AutogradPrivateUse1)
def batch_norm_decomposition(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> Tensor:
    if input.numel() == 0:
        out = input.clone()
        if weight is not None:
            out *= weight[0]
        if bias is not None:
            out += bias[0]
        return out
    return aten._batch_norm_impl_index.default(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enabled,
    )[0]

def npu_convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    if not output_mask[2]:
        return NotImplemented
    grad_bias = torch.ops.aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, grad_weight, _ = torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], output_mask[1], False],
    )
    return (grad_inp, grad_weight, grad_bias)
 
def npu__softmax_backward_data(
    grad_output: torch.Tensor,
    output: torch.Tensor,
    dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    new_grad_output = grad_output * output
    sum_new_grad = torch.sum(new_grad_output, dim=dim, keepdim=True)
    grad_input = new_grad_output - output * sum_new_grad
    # grad_input = inductor_prims.fma(-output, sum_new_grad, new_grad_output)

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()

    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input.contiguous()

def npu_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon=1e-6
) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
    output = (x * rsqrt * weight).to(dtype)
    return output, rsqrt

def npu_rms_norm_backward(grad_output: torch.Tensor, 
                      x: torch.Tensor, 
                      weight: torch.Tensor, 
                      rsqrt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = (grad_output * weight - x * rsqrt * (grad_output * weight * x * rsqrt).mean(-1, keepdim=True)) * rsqrt
    dgamma = (grad_output * x * rsqrt).sum(0, keepdim=False)
    return dx, dgamma

def npu_swiglu(x, dim=-1):
    x = torch.chunk(x, 2, dim=dim)
    return F.silu(x[0]) * x[1]

def npu_swiglu_backward(grad_output, x, dim=-1):
    x0, x1 = torch.chunk(x, 2, dim=dim)

    # 计算 x0 的梯度
    sigmoid_x0 = torch.sigmoid(x0)
    silu_grad = sigmoid_x0 * (1 + x0 * (1 - sigmoid_x0))  # SiLU 的导数
    grad_x0 = grad_output * x1 * silu_grad

    # 计算 x1 的梯度
    grad_x1 = grad_output * F.silu(x0)
    grad_x = torch.cat([grad_x0, grad_x1], dim=dim)
    return grad_x

def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
    
def npu_rotary_mul(t, cos_, sin_):
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return t

def npu_rotary_mul_backward(grad_output, t, cos_, sin_):
    rotated_t = _rotate_half(t)
    grad_t = cos_ * grad_output
    grad_rotated_part = grad_output * sin_
    a, b = torch.chunk(grad_rotated_part, 2, dim=-1)
    grad_rotated_t = torch.cat((b, -a), dim=-1)
    grad_t = grad_t + grad_rotated_t

    grad_cos = t * grad_output
    grad_sin = rotated_t * grad_output

    return grad_t, grad_cos, grad_sin

def gelu(a, approximate: str = "none"):
    """
    Reference implementation of torch.nn.functional.gelu
    """
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
    kKappa = 0.044715
    a_cube = a * a * a
    inner = kBeta * (a + kKappa * a_cube)
    return 0.5 * a * (1 + torch.tanh(inner))

def gelu_backward(grad: Tensor, self: Tensor, approximate: str = "none"):
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
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

    return grad * (left_derivative + right_derivative)


def native_dropout(tensor_input, p, train):
    if train and p != 0:
        return torch.ops.npu._npu_dropout(tensor_input, p)
    return (tensor_input, torch.ones_like(tensor_input, dtype=torch.bool))


def native_dropout_backward(grad_output, mask, scale):
    p = 1 if scale == 0 else (1 - 1 / scale)
    r = torch.ops.npu.npu_dropout_backward(grad_output, mask, p)
    return r


inductor_decomp.register_decomposition(torch.ops.aten.convolution_backward)(npu_convolution_backward)
inductor_decomp.register_decomposition(torch.ops.aten._softmax_backward_data.default)(npu__softmax_backward_data)
inductor_decomp.register_decomposition(torch.ops.aten.gelu.default)(gelu)
inductor_decomp.register_decomposition(torch.ops.aten.gelu_backward.default)(gelu_backward)
inductor_decomp.register_decomposition(torch.ops.npu.npu_rms_norm.default)(npu_rms_norm)
inductor_decomp.register_decomposition(torch.ops.npu.npu_rms_norm_backward.default)(npu_rms_norm_backward)
inductor_decomp.register_decomposition(torch.ops.aten.native_dropout.default)(native_dropout)
inductor_decomp.register_decomposition(torch.ops.aten.native_dropout_backward.default)(native_dropout_backward)


# inductor_decomp.register_decomposition(torch.ops.npu.npu_swiglu.default)(npu_swiglu)
# inductor_decomp.register_decomposition(torch.ops.npu.npu_swiglu_backward.default)(npu_swiglu_backward)
# inductor_decomp.register_decomposition(torch.ops.npu.npu_rotary_mul.default)(npu_rotary_mul)
# inductor_decomp.register_decomposition(torch.ops.npu.npu_rotary_mul_backward.default)(npu_rotary_mul_backward)