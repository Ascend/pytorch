import os
import functools
from typing import Optional, Tuple
import torch
import torch._ops
from torch import Tensor
from torch._inductor import decomposition as inductor_decomp
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath, register_decomposition
from torch._C import DispatchKey
from torch._decomp import remove_decompositions
from torch._prims_common.wrappers import out_wrapper
import torch.nn.functional as F
from .lowering_common import add_overload
from .ascend_npu_ir.ascend_npu_ir import config as anir_config
from .lowering_common import run_once

aten = torch.ops.aten
npu = torch.ops.npu


@run_once
def _register_shared_decompositions():
    @register_decomposition([aten.expm1])
    def expm1(x):
        tensor = torch.exp(x) - torch.ones_like(x)
        return tensor

def _register_triton_decompositions():
    from .config import is_ascend950, enable_fast_gelu
    from .lowering import _add_overload
    DECOMPOSITION_OVERLOAD_OP = [
        aten.nll_loss_forward,
        aten.nll_loss_backward,
        aten._log_softmax_backward_data,
        aten.erfc,
        aten.gelu,
        aten.native_layer_norm,
        aten.slice_backward,
    ]

    if is_ascend950:
        DECOMPOSITION_OVERLOAD_OP.append(aten.max_pool2d_with_indices)

    def _register_npu_triton_decompositions():
        overload_op_set = set()
        add_overload(DECOMPOSITION_OVERLOAD_OP, overload_op_set)

        for op in overload_op_set:
            if (op in decompositions):
                del decompositions[op]

        @register_decomposition([aten.erfc])
        def erfc(x):
            tensor = torch.ones_like(x) - torch.erf(x)
            return tensor

        if enable_fast_gelu:
            @register_decomposition([aten.gelu])
            def gelu(x):
                two_sqrt_2_over_pi = 1.5957691216057308
                coeff = 0.044715
                x_cubed = x * x * x
                z = two_sqrt_2_over_pi * (x + coeff * x_cubed)
                sigmoid_z = torch.sigmoid(z)
                result = x * sigmoid_z
                return result

    _register_npu_triton_decompositions()

def _register_mlir_dvm_decompositions():
    exclude_list = anir_config.decomps_to_exclude_npu
    if os.getenv("TORCHINDUCTOR_NPU_BACKEND", "default") == "mlir":
        exclude_list.append(torch.ops.aten._safe_softmax)
    remove_decompositions(inductor_decomp.decompositions, exclude_list)

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

    # Constants from cann/ops-nn gelu / gelu_grad
    _GELU_BETA = 1.595769121605730711759  # sqrt(8/pi)
    _GELU_KAPPA = 0.044715
    _GELU_AN = -0.0713548162726002527220  # -BETA * KAPPA
    _GELU_A3 = 0.2140644488178007  # BETA * 3 * KAPPA
    _M_SQRT1_2 = 0.70710678118654752440
    _INV_SQRT_2PI = 0.3989422804

    # AscendC Erf PADE (adv_api/detail/math/erf/erf_common_impl.h)
    _ERF_CLIP = 3.92
    _ERF_P5 = 0.053443748819
    _ERF_P4 = 0.75517016694e1
    _ERF_P3 = 0.10162808918e3
    _ERF_P2 = 0.13938061484e4
    _ERF_P1 = 0.50637915060e4
    _ERF_P0 = 0.29639384698e5
    _ERF_Q4 = 0.31212858877e2
    _ERF_Q3 = 0.39856963806e3
    _ERF_Q2 = 0.30231248150e4
    _ERF_Q1 = 0.13243365831e5
    _ERF_Q0 = 0.26267224157e5

    def _npu_use_compatible_gelu_v2() -> bool:
        try:
            from torch_npu.npu import are_compatible_impl_enabled

            return are_compatible_impl_enabled()
        except Exception:
            return False

    def _gelu_use_tanh_approx(approximate: str) -> bool:
        if approximate == "tanh":
            return True
        if approximate == "none":
            return not _npu_use_compatible_gelu_v2()
        raise RuntimeError(
            f"approximate argument must be either none or tanh, but got {approximate!r}."
        )

    def _erf_clip_fp32(x: Tensor) -> Tensor:
        """AscendC ErfClip: Mins(x, 3.92) then Maxs(..., -3.92)."""
        x = torch.clamp_max(x, _ERF_CLIP)
        x = torch.clamp_min(x, -_ERF_CLIP)
        return x

    def _erf_compute_p_fp32(x: Tensor, x2: Tensor) -> Tensor:
        """AscendC ErfComputeP: P(x) Horner (Muls/Adds/Mul)."""
        t = x2 * _ERF_P5
        t = t + _ERF_P4
        t = x2 * t
        t = t + _ERF_P3
        t = x2 * t
        t = t + _ERF_P2
        t = x2 * t
        t = t + _ERF_P1
        t = x2 * t
        t = t + _ERF_P0
        return x * t

    def _erf_compute_q_fp32(x2: Tensor) -> Tensor:
        """AscendC ErfComputeQ: Q(x) Horner (Adds/Mul)."""
        t = x2 + _ERF_Q4
        t = x2 * t
        t = t + _ERF_Q3
        t = x2 * t
        t = t + _ERF_Q2
        t = x2 * t
        t = t + _ERF_Q1
        t = x2 * t
        t = t + _ERF_Q0
        return t

    def _erf_pade_fp32(x: Tensor) -> Tensor:
        """AscendC ErfCompute: clip then P(x)/Q(x)."""
        x = _erf_clip_fp32(x)
        x2 = x * x
        p = _erf_compute_p_fp32(x, x2)
        q = _erf_compute_q_fp32(x2)
        return p / q

    def erf(a: Tensor) -> Tensor:
        """Match AscendC ErfImpl; fp16: cast up CAST_NONE, compute in fp32, cast down."""
        orig_dtype = a.dtype
        if orig_dtype != torch.float32:
            a = a.to(torch.float32)
        out = _erf_pade_fp32(a)
        if orig_dtype != torch.float32:
            out = out.to(orig_dtype)
        return out

    def gelu(a, approximate: str = "none"):
        """Match eager gelu; see cann/ops-nn gelu decomp."""
        orig_dtype = a.dtype
        if orig_dtype != torch.float32:
            a = a.to(torch.float32)
        if _gelu_use_tanh_approx(approximate):
            a_cube = a * a * a
            out = a / (1.0 + torch.exp(-_GELU_BETA * (a + _GELU_KAPPA * a_cube)))
        else:
            # GeluV2ErfPost: (1 + erf) * (0.5 * x)
            out = (1.0 + _erf_pade_fp32(a * _M_SQRT1_2)) * (0.5 * a)
        if orig_dtype != torch.float32:
            out = out.to(orig_dtype)
        return out

    def gelu_backward(grad: Tensor, self: Tensor, approximate: str = "none"):
        """Match eager gelu_backward; see cann/ops-nn gelu_grad decomp."""
        orig_dtype = grad.dtype
        if orig_dtype != torch.float32:
            grad = grad.to(torch.float32)
            self = self.to(torch.float32)
        if _gelu_use_tanh_approx(approximate):
            x_sq = self * self
            px = torch.exp((-_GELU_BETA + _GELU_AN * x_sq) * self)
            res0 = (_GELU_BETA + _GELU_A3 * x_sq) * self
            div = 1.0 / (1.0 + px)
            resp = px * div * res0 * div
            resp = torch.where(torch.isnan(resp), torch.zeros_like(resp), resp)
            out = grad * (resp + div)
        else:
            cdf = 0.5 * (1.0 + _erf_pade_fp32(self * _M_SQRT1_2))
            pdf = _INV_SQRT_2PI * torch.exp(self * self * -0.5)
            out = grad * (cdf + self * pdf)
        if orig_dtype != torch.float32:
            out = out.to(orig_dtype)
        return out


    register_decomposition(torch.ops.aten.convolution_backward)(npu_convolution_backward)
    register_decomposition(torch.ops.aten._softmax_backward_data.default)(npu__softmax_backward_data)
    register_decomposition(torch.ops.aten.erf.default)(erf)
    register_decomposition(torch.ops.aten.gelu.default)(gelu)
    register_decomposition(torch.ops.aten.gelu_backward.default)(gelu_backward)
    # register_decomposition(torch.ops.npu.npu_rms_norm.default)(npu_rms_norm)
    # register_decomposition(torch.ops.npu.npu_rms_norm_backward.default)(npu_rms_norm_backward)
    # register_decomposition(torch.ops.npu.npu_swiglu.default)(npu_swiglu)
    # register_decomposition(torch.ops.npu.npu_swiglu_backward.default)(npu_swiglu_backward)
    # register_decomposition(torch.ops.npu.npu_rotary_mul.default)(npu_rotary_mul)
    # register_decomposition(torch.ops.npu.npu_rotary_mul_backward.default)(npu_rotary_mul_backward)


def _register_npu_inductor_decompositions(backend=None):
    if backend == "triton":
        _register_triton_decompositions()
    elif backend in ["mlir_dvm"]:
        _register_mlir_dvm_decompositions()
