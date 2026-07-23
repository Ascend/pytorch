from typing import Optional, Tuple
import torch
import torch._ops
from torch import Tensor
from torch._inductor import decomposition as inductor_decomp
from torch._inductor.decomposition import decompositions, register_decomposition
from torch._C import DispatchKey
from torch._decomp import decomposition_table, remove_decompositions
from torch._prims_common.wrappers import out_wrapper  # noqa: F401
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
    from .config import is_ascend950
    from .lowering import _add_overload  # noqa: F401
    DECOMPOSITION_OVERLOAD_OP = [
        aten.nll_loss_forward,
        aten.nll_loss_backward,
        aten._log_softmax_backward_data,
        aten.addmm,
        aten.gelu,
        aten.native_layer_norm,
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

    def npu_rms_norm_backward(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        rsqrt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = (grad_output * weight - x * rsqrt * (grad_output * weight * x * rsqrt).mean(-1, keepdim=True)) * rsqrt
        # gamma is broadcast over every dim except the last, so its gradient must
        # reduce over all leading dims. sum(0) only handles rank-2 input and
        # leaves the middle dims un-reduced for rank>2 (shape mismatch vs the op).
        reduce_dims = tuple(range(grad_output.dim() - 1))
        dgamma = (grad_output * x * rsqrt).sum(reduce_dims, keepdim=False)
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

    register_decomposition(torch.ops.aten.convolution_backward)(npu_convolution_backward)
    register_decomposition(torch.ops.aten._softmax_backward_data.default)(npu__softmax_backward_data)
    register_decomposition(torch.ops.aten.gelu.default)(gelu)
    register_decomposition(torch.ops.aten.gelu_backward.default)(gelu_backward)
    # register_decomposition(torch.ops.npu.npu_rms_norm.default)(npu_rms_norm)
    # register_decomposition(torch.ops.npu.npu_rms_norm_backward.default)(npu_rms_norm_backward)
    # register_decomposition(torch.ops.npu.npu_swiglu.default)(npu_swiglu)
    # register_decomposition(torch.ops.npu.npu_swiglu_backward.default)(npu_swiglu_backward)
    # register_decomposition(torch.ops.npu.npu_rotary_mul.default)(npu_rotary_mul)
    # register_decomposition(torch.ops.npu.npu_rotary_mul_backward.default)(npu_rotary_mul_backward)


# ---------------------------------------------------------------------------
# triton_experimental backend decomposition / dispatcher / SDPA overrides.
#
# Physically these used to live in
# ``torch_npu/_inductor/triton_experimental/decomposition.py``. They are grouped
# here alongside the other per-backend decomposition registrars
# (``_register_triton_decompositions`` / ``_register_mlir_dvm_decompositions``);
# the loader ``_load_triton_experimental_backend`` calls
# ``_register_triton_experimental_decompositions`` directly, so nothing runs unless
# the triton_experimental backend is actually selected.
# ---------------------------------------------------------------------------

# Keeps the PythonDispatcher Library alive for the process lifetime. Created lazily
# (call-time, not import-time) so importing this shared module under other backends
# never registers these aten dispatcher impls.
_te_python_dispatcher_lib = None


def npu_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                     is_causal=False, scale=None, enable_gqa=False):
    """Python equivalent of ScaledDotProductAttentionKernelNpuOpApi.cpp (V2R5+)."""
    ATTENMASK_LIMIT = 2048
    N_LIMIT = 2048
    D_LIMIT = 512
    BNSD_DIM = 4
    TOKEN_MAX = 2147483647
    LEFT_UP_CAUSAL = 2

    def _convert_boolean_attn_mask(query, attn_mask, is_causal):
        if attn_mask is None and not is_causal:
            return None
        if is_causal:
            return torch.triu(
                torch.ones(ATTENMASK_LIMIT, ATTENMASK_LIMIT, dtype=torch.bool, device=query.device), 1
            )
        return torch.logical_not(attn_mask)


    def _convert_boolean_attn_mask_math(attn_mask, dtype):
        if attn_mask is None:
            return None
        if attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=dtype)
            new_attn_mask.masked_fill_(attn_mask.logical_not(), float('-inf'))
            return new_attn_mask
        return attn_mask


    def _calculate_scale(query, scale):
        if scale is not None:
            return scale
        return 1.0 / (query.size(-1) ** 0.5)

    # Path 1: FlashAttention via npu_fusion_attention
    if ((query.dtype in (torch.float16, torch.bfloat16, torch.float32)) and
            ((attn_mask is not None and attn_mask.dtype == torch.bool) or attn_mask is None) and
            query.dim() == BNSD_DIM and key.dim() == BNSD_DIM and value.dim() == BNSD_DIM and
            query.size(1) <= N_LIMIT and query.size(3) <= D_LIMIT and key.size(1) <= N_LIMIT and
            query.size(1) % key.size(1) == 0 and query.size(1) // key.size(1) > 0):
        # attn_mask must be dim 2 or 4 for FA path
        if attn_mask is not None and attn_mask.dim() not in (2, 4):
            atten_mask_math = _convert_boolean_attn_mask_math(attn_mask, query.dtype)
            return torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, atten_mask_math, dropout_p, is_causal, None, scale=scale, enable_gqa=enable_gqa
            )[0]

        atten_mask = _convert_boolean_attn_mask(query, attn_mask, is_causal)
        head_num = query.size(1)
        input_layout = "BNSD"
        input_scale = _calculate_scale(query, scale)
        keep_prob = 1.0 - dropout_p
        next_tockens = 0 if is_causal else TOKEN_MAX
        sparse_mode = LEFT_UP_CAUSAL if is_causal else 0

        output = torch.ops.npu.npu_fusion_attention_v3(
            query, key, value, head_num, input_layout,
            pse=None, padding_mask=None, atten_mask=atten_mask,
            scale=input_scale, keep_prob=keep_prob,
            pre_tockens=TOKEN_MAX, next_tockens=next_tockens,
            inner_precise=0, prefix=None,
            actual_seq_qlen=None, actual_seq_kvlen=None,
            sparse_mode=sparse_mode, gen_mask_parallel=True, sync=False
        )
        return output[0]

    # Path 2: Math fallback
    atten_mask_math = _convert_boolean_attn_mask_math(attn_mask, query.dtype)
    return torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, atten_mask_math, dropout_p, is_causal, None, scale=scale, enable_gqa=enable_gqa
    )[0]


def apply_inductor_scaled_dot_product_attention_patch():
    torch.nn.functional.scaled_dot_product_attention = npu_scaled_dot_product_attention


@run_once
def disable_implicit_decomposition():
    '''
    Disable implicit decomposition of some aten ops to avoid poor performance.
    '''
    disable_aten_ops = [
        'aten.upsample_nearest1d.vec', 'aten.upsample_nearest1d.default',
        'aten.upsample_nearest2d.vec', 'aten.upsample_nearest2d.default',
        'aten.upsample_nearest3d.vec', 'aten.upsample_nearest3d.default',
        'aten.upsample_bilinear2d.vec', 'aten.upsample_bilinear2d.default',
    ]

    for op_override in decomposition_table.keys():
        if str(op_override) in disable_aten_ops:
            op_override.py_kernels.pop(DispatchKey.Autograd, None)
            op_override.py_kernels.pop(DispatchKey.CompositeImplicitAutograd, None)


def _override_matmul_should_fold_for_npu():
    """Force `aten.matmul([S,B,E], [E,N])` to fold into a 2D `mm` on NPU.

    Upstream should_fold returns False for a non-contiguous 3D LHS (seq-first
    attention query.transpose(0,1)), routing to the pathological bmm branch that
    broadcasts the 2D weight to [S,E,N] then runs an S-batch BMM (CLIP QKV ~24x
    slower: 775us bmm vs 32us mm). Folding costs only a contiguous() copy, so fold
    any (>=3D LHS) x (<=2D RHS). NOT a blanket True: the fold branch reshapes the
    LHS to 2D and needs t2.ndim <= 2, so true batched matmuls must keep upstream.
    Patched on the module attribute (the decomposition calls it by global name).
    """
    import torch._decomp.decompositions as _decomp_mod

    orig_should_fold = _decomp_mod.should_fold

    # Idempotent: don't re-wrap if already patched (re-import / re-init).
    if getattr(orig_should_fold, "_npu_fold_3d_2d", False):
        return

    def should_fold(tensor1, tensor2, is_out):
        t1, t2 = (
            (tensor1, tensor2)
            if tensor1.ndim >= tensor2.ndim
            else (tensor2, tensor1)
        )
        # Any (>=3D LHS) x (<=2D RHS): fold regardless of contiguity (the fold
        # branch's own invariant), avoiding the upstream broadcast+bmm path.
        if t1.ndim >= 3 and t2.ndim <= 2:
            return True
        return orig_should_fold(tensor1, tensor2, is_out)

    should_fold._npu_fold_3d_2d = True
    _decomp_mod.should_fold = should_fold


def _override_softmax_backward_decomp_no_fma():
    """Replace inductor's ``_softmax_backward_data`` decomposition so it avoids
    ``inductor_prims.fma`` (materialised as prims.fma on NPU). Use the equivalent
    ``new_grad_output - output * sum_new_grad`` so the kernel stays on mul/sub."""
    from torch._inductor.decomposition import (
        decompositions as _ind_decomps,
        pw_cast_for_opmath,
    )

    @pw_cast_for_opmath
    def _softmax_backward_data_no_fma(
        grad_output: torch.Tensor,
        output: torch.Tensor,
        dim: int,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        new_grad_output = grad_output * output
        sum_new_grad = torch.sum(new_grad_output, dim=dim, keepdim=True)
        grad_input = new_grad_output - output * sum_new_grad
        if grad_output.dtype != input_dtype:
            grad_input = grad_input.to(input_dtype)
        return grad_input.contiguous()

    _ind_decomps[aten._softmax_backward_data.default] = _softmax_backward_data_no_fma


def _override_gelu_decomp():
    from torch._inductor.decomposition import decompositions as _ind_decomps

    def _gelu(x: torch.Tensor, approximate: str = "none"):
        two_sqrt_2_over_pi = 1.5957691216057308
        coeff = 0.044715
        x_cubed = x * x * x
        z = two_sqrt_2_over_pi * (x + coeff * x_cubed)
        sigmoid_z = torch.sigmoid(z)
        result = x * sigmoid_z
        return result

    def _gelu_backward(grad: torch.Tensor, self: torch.Tensor, approximate: str = "none"):
        two_sqrt_2_over_pi = 1.5957691216057308
        coeff = 0.044715
        x_sq = self * self
        x_cubed = x_sq * self
        z = two_sqrt_2_over_pi * (self + coeff * x_cubed)
        sigmoid_z = torch.sigmoid(z)
        # dz/dx = two_sqrt_2_over_pi * (1 + 3*coeff*x^2)
        z_derivative = two_sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_sq)
        # d/dx [x * sigmoid(z)] = sigmoid(z) + x * sigmoid(z)*(1 - sigmoid(z)) * dz/dx
        sigmoid_derivative = sigmoid_z * (1.0 - sigmoid_z)
        result_derivative = sigmoid_z + self * sigmoid_derivative * z_derivative
        return grad * result_derivative

    _ind_decomps[aten.gelu.default] = _gelu
    _ind_decomps[aten.gelu_backward.default] = _gelu_backward


def _override_rms_norm_decomp():
    from torch._inductor.decomposition import decompositions as _ind_decomps

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
        # gamma is broadcast over every dim except the last, so its gradient must
        # reduce over all leading dims. sum(0) only handles rank-2 input and
        # leaves the middle dims un-reduced for rank>2 (shape mismatch vs the op).
        reduce_dims = tuple(range(grad_output.dim() - 1))
        dgamma = (grad_output * x * rsqrt).sum(reduce_dims, keepdim=False)
        return dx, dgamma

    _ind_decomps[npu.npu_rms_norm.default] = npu_rms_norm
    _ind_decomps[npu.npu_rms_norm_backward.default] = npu_rms_norm_backward


def _override_native_dropout_decomp():
    """Override aten.native_dropout / native_dropout_backward for NPU. The core
    aten decomposition produces a bool mask, but this CANN's dropout backward
    (aclnnDropoutDoMask) rejects DT_BOOL and needs the packed uint8 bitmask from
    npu._npu_dropout. Forward calls _npu_dropout for training p∈(0,1); backward
    routes packed masks to npu.npu_dropout_backward and bool masks to the math
    form (grad * mask * scale)."""
    from torch._inductor.decomposition import decompositions as _ind_decomps
    from torch._decomp.decompositions_for_rng import extra_random_decomps

    def native_dropout(input: torch.Tensor, p: float, train: Optional[bool]):
        dropout_train = True if train is None else train
        if p == 0 or not dropout_train:
            mask = torch.ones_like(input, dtype=torch.bool)
            return input.clone(), mask
        if p == 1:
            output = torch.zeros_like(input)
            mask = torch.zeros_like(input, dtype=torch.bool)
            return output, mask
        # _npu_dropout returns (result, packed uint8 mask)
        return torch.ops.npu._npu_dropout(input, p)

    def native_dropout_backward(grad_output: torch.Tensor, mask: torch.Tensor, scale: float):
        # aten.native_dropout_backward's third arg is scale = 1 / (1 - p),
        # while npu.npu_dropout_backward expects p. Fast paths match
        # NativeDropoutKernelNpuOpApi.cpp.
        p = 1.0 if scale == 0.0 else (1.0 - 1.0 / scale)
        if p == 0:
            return grad_output.clone()
        if p == 1:
            return torch.zeros_like(grad_output)
        if tuple(mask.shape) == tuple(grad_output.shape):
            return grad_output * mask.to(dtype=grad_output.dtype) * scale
        return torch.ops.npu.npu_dropout_backward(grad_output, mask, p)

    _ind_decomps[aten.native_dropout.default] = native_dropout
    _ind_decomps[aten.native_dropout_backward.default] = native_dropout_backward
    # select_decomp_table() uses fast_random_decomps() =
    # {**decompositions, **extra_random_decomps}. Since extra_random_decomps
    # contains aten.native_dropout, it would otherwise override _ind_decomps and
    # keep emitting inductor_random_default + bool gt masks.
    extra_random_decomps[aten.native_dropout.default] = native_dropout


def _register_triton_experimental_decompositions():
    """Install all triton_experimental decomposition-table overrides and prune the
    exclusion list. Called directly by ``_load_triton_experimental_backend``.

    Mirrors the original in-order sequence: install the aten dispatcher impls,
    remove the fallback-preferred decompositions, clear the ``fast_random_decomps``
    lru cache (materialised with the old table), then install each override and
    clear the cache again so ``select_decomp_table`` rebuilds from the new table.
    """
    npu_decomps_to_exclude = [
        aten.convolution_backward,
        aten.native_dropout,
        aten.max_pool2d_with_indices,
        aten.max_pool2d_with_indices_backward,
        aten.embedding,
        aten.embedding_dense_backward,
    ]
    # On A5 (910_95), let these ops go through decomposition instead of
    # falling back, so drop them from the exclusion list.
    a5_decomps_to_include = [
        aten.embedding,
        aten.embedding_dense_backward,
    ]
    from .triton_experimental import device_props as _device_props
    if _device_props.is_a5():
        for op in a5_decomps_to_include:
            npu_decomps_to_exclude.remove(op)

    disable_implicit_decomposition()
    apply_inductor_scaled_dot_product_attention_patch()
    remove_decompositions(decompositions, npu_decomps_to_exclude)

    # Also remove from fast_random_decomps cache (used by select_decomp_table
    # when config.fallback_random=False). The lru_cache means the dict may
    # already be materialized with the old decompositions.
    from torch._inductor.decomposition import fast_random_decomps
    fast_random_decomps.cache_clear()

    _override_matmul_should_fold_for_npu()
    _override_softmax_backward_decomp_no_fma()
    _override_gelu_decomp()
    _override_rms_norm_decomp()
    _override_native_dropout_decomp()
    fast_random_decomps.cache_clear()
