import torch
from torch._decomp import get_decompositions, remove_decompositions
from torch._decomp.decompositions import compute_only_pw_cast_for_opmath
from torch._inductor import decomposition as inductor_decomp
from torch._inductor.decomposition import pw_cast_for_opmath

from . import config as dvm_config

aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized

decomps_to_exclude_npu = [
    aten._batch_norm_no_update,
    aten._batch_norm_with_update,
    aten._batch_norm_with_update_functional,
    aten._log_softmax,
    aten._log_softmax_backward_data,
    aten.batch_norm_backward,
    aten.convolution_backward,
    aten.embedding,
    aten.embedding_backward,
    aten.embedding_dense_backward,
    aten.index_select,
    aten.gelu.default,
    aten.gelu_backward.default,
    aten.grid_sampler_2d,
    aten.grid_sampler_2d_backward,
    aten.linalg_vector_norm,
    aten.max_pool2d_with_indices,
    aten.max_pool2d_with_indices_backward,
    aten.native_batch_norm,
    aten.nll_loss2d_backward,
    aten.nll_loss2d_forward,
    aten.nll_loss_backward,
    aten.nll_loss_forward,
    aten.reflection_pad2d,
    aten.reflection_pad2d_backward,
    aten.silu.default,
    aten.silu_backward.default,
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

cia_decomps_to_exclude_npu = [
    aten.silu_backward.default,
]

FP32_MIN_V2 = -8.8
FP32_MAX_V2 = 8.8
DOUBLE_X = 2.0


@pw_cast_for_opmath
def tanh(a):
    """
    y = (exp(2x) - 1) / (exp(2x) + 1)
    with x clipped to [-8.8, 8.8] in float32 before multiply-by-2.
    """
    x = torch.clamp(a, min=FP32_MIN_V2, max=FP32_MAX_V2)
    x2 = x * DOUBLE_X
    e2x = torch.exp(x2)
    return (e2x - 1.0) / (e2x + 1.0)


@pw_cast_for_opmath
def sigmoid(a: torch.Tensor) -> torch.Tensor:
    return aten.reciprocal(1.0 + torch.exp(torch.neg(a)))


@pw_cast_for_opmath
def silu(a: torch.Tensor) -> torch.Tensor:
    return a / (1.0 + torch.exp(torch.neg(a)))


@pw_cast_for_opmath
def silu_backward(grad: torch.Tensor, self: torch.Tensor) -> torch.Tensor:
    sigmoid = aten.reciprocal(1.0 + torch.exp(torch.neg(self)))
    return grad * (sigmoid * (1.0 + (1.0 - sigmoid) * self))


@compute_only_pw_cast_for_opmath
def _npu_rms_norm_fp32(
    x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_dims = tuple(range(x.ndim - gamma.ndim, x.ndim))
    rstd = torch.rsqrt((x * x).mean(dim=norm_dims, keepdim=True) + epsilon)
    y = (x * rstd) * gamma
    return y, rstd


def npu_rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    y, rstd = _npu_rms_norm_fp32(x, gamma, epsilon)
    return y.to(x.dtype), rstd


@compute_only_pw_cast_for_opmath
def _npu_rms_norm_backward_fp32(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_dims = tuple(range(x.ndim - gamma.ndim, x.ndim))
    batch_dims = tuple(range(x.ndim - gamma.ndim))
    x_normalized = x * rstd
    grad_scaled = grad_output * gamma
    correction = (grad_scaled * x_normalized).mean(dim=norm_dims, keepdim=True)
    dx = (grad_scaled - x_normalized * correction) * rstd
    dgamma = grad_output * x_normalized
    if batch_dims:
        dgamma = dgamma.sum(dim=batch_dims)
    return dx, dgamma


def npu_rms_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx, dgamma = _npu_rms_norm_backward_fp32(grad_output, x, gamma, rstd)
    return dx.to(x.dtype), dgamma


def _disable_cia_decompositions():
    """Keep FunctionalTensorMode from expanding ops before DVM decompositions."""
    dispatch_key = torch._C.DispatchKey.CompositeImplicitAutograd

    def preserve_for_explicit_decomposition(*_args, **_kwargs):
        return NotImplemented

    for op in cia_decomps_to_exclude_npu:
        op.py_kernels.pop(dispatch_key, None)
        op.py_impl(dispatch_key)(preserve_for_explicit_decomposition)


# Constants from cann/ops-nn gelu / gelu_grad
_GELU_BETA = 1.595769121605730711759  # sqrt(8/pi)
_GELU_KAPPA = 0.044715
_GELU_AN = -0.0713548162726002527220  # -BETA * KAPPA
_GELU_A3 = 0.2140644488178007  # BETA * 3 * KAPPA
_M_SQRT1_2 = 0.70710678118654752440
_INV_SQRT_2PI = 0.3989422804

# AscendC Erf PADE
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


def _erf_clip_fp32(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp_max(x, _ERF_CLIP)
    x = torch.clamp_min(x, -_ERF_CLIP)
    return x


def _erf_compute_p_fp32(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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


def _erf_compute_q_fp32(x2: torch.Tensor) -> torch.Tensor:
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


def _erf_pade_fp32(x: torch.Tensor) -> torch.Tensor:
    x = _erf_clip_fp32(x)
    x2 = x * x
    p = _erf_compute_p_fp32(x, x2)
    q = _erf_compute_q_fp32(x2)
    return p / q


@pw_cast_for_opmath
def gelu(a: torch.Tensor, approximate: str = "none"):
    """Match eager gelu; DVM path uses resp==resp NaN clear in backward."""
    if _gelu_use_tanh_approx(approximate):
        a_cube = a * a * a
        out = a / (1.0 + torch.exp(-_GELU_BETA * (a + _GELU_KAPPA * a_cube)))
    else:
        out = (1.0 + _erf_pade_fp32(a * _M_SQRT1_2)) * (0.5 * a)
    return out


@pw_cast_for_opmath
def gelu_backward(grad: torch.Tensor, self: torch.Tensor, approximate: str = "none"):
    """Match eager gelu_backward; NaN clear via resp==resp (CANN Compare EQ)."""
    if _gelu_use_tanh_approx(approximate):
        x_sq = self * self
        px = torch.exp((-_GELU_BETA + _GELU_AN * x_sq) * self)
        res0 = (_GELU_BETA + _GELU_A3 * x_sq) * self
        div = 1.0 / (1.0 + px)
        resp = px * div * res0 * div
        resp = torch.where(resp == resp, resp, torch.zeros_like(resp))
        out = grad * (resp + div)
    else:
        cdf = 0.5 * (1.0 + _erf_pade_fp32(self * _M_SQRT1_2))
        pdf = _INV_SQRT_2PI * torch.exp(self * self * -0.5)
        out = grad * (cdf + self * pdf)
    return out


dvm_decompositions = {
    aten.sigmoid.default: sigmoid,
    aten.silu.default: silu,
    aten.silu_backward.default: silu_backward,
    aten.gelu_backward.default: gelu_backward,
    aten.gelu.default: gelu,
    aten.tanh.default: tanh,
    torch.ops.npu.npu_rms_norm.default: npu_rms_norm,
    torch.ops.npu.npu_rms_norm_backward.default: npu_rms_norm_backward,
}


_dvm_inductor_decomp_patched = False


def _register_inductor_decomposition_safe(overloads, fn):
    """Register a custom Inductor decomposition; ignore duplicate registration."""
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
    """Patch Inductor decomposition for DVM paths (idempotent)."""
    global _dvm_inductor_decomp_patched
    if _dvm_inductor_decomp_patched:
        return
    from ..decomposition import _override_matmul_should_fold_for_npu

    _override_matmul_should_fold_for_npu()
    _disable_cia_decompositions()
    remove_decompositions(inductor_decomp.decompositions, decomps_to_exclude_npu)
    for op, fn in dvm_decompositions.items():
        if op is not torch.ops.npu.npu_rms_norm_backward.default:
            _register_inductor_decomposition_safe([op], fn)
    for op, fn in get_decompositions(dvm_config.enable_decomp_list).items():
        inductor_decomp.decompositions.setdefault(op, fn)
    for op, fn in dvm_decompositions.items():
        if (
            op in dvm_config.enable_decomp_list
            or op.overloadpacket in dvm_config.enable_decomp_list
        ):
            inductor_decomp.decompositions[op] = fn
    for table in (
        inductor_decomp.decompositions,
        inductor_decomp.extra_random_decomps,
    ):
        remove_decompositions(table, dvm_config.disable_decomp_list)
    inductor_decomp.fast_random_decomps.cache_clear()
    _dvm_inductor_decomp_patched = True
