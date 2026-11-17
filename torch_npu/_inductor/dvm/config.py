"""Shared configuration for DVM Inductor integration."""

import os

import torch
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._inductor.fx_passes.control_dependencies import control_deps

from torch_npu._inductor.config import is_ascend950

aten = torch.ops.aten
prims = torch.ops.prims

# Run post-launch DVM debug checks.
debug_mode = False
# Emit standalone FX regression cases for DVM-fused graphs.
dump_fx_test = False
# Enable DVM cat fusion only when explicitly requested.
enable_cat = False
# Enable DVM permute fusion; permute lowering can emit reinterpret_tensor views.
enable_fusion_permute = False
# Configure operator packets/overloads before the DVM backend is loaded.
# Disabling takes precedence when an operator occurs in both lists.
disable_decomp_list = []
enable_decomp_list = []
# C310 VF fusion: 0 off, 1 validates shapes, 2 assumes equal shapes.
vf_fusion = 2
# View-load fusion: 0 off, 1 requires a unit trailing stride,
# 2 allows arbitrary strides for vector kernels; mix uses 0, spec/split/concat at most 1.
view_fusion_level = 1
# Use DVM-specific fusion rules that prevent post-reduction fusion.
disable_post_reduce_fusion = False
# Enable DVM matmul fusion for mm, bmm, addmm, and baddbmm.
enable_matmul_fusion = (
    os.environ.get("INDUCTOR_DVM_ENABLE_MATMUL_FUSION", "0") == "1"
)
# Avoid extra casts in the pre-A5 aclnnAddmm path by keeping mm + add
# separate by default unless matmul templates are enabled.
# A5 keeps Inductor's default.
disable_addmm_fusion = not is_ascend950
# Optional callback (n, m, k) for (..., m, k) @ (..., k, n).
# Returning False vetoes fusion; otherwise the existing fusion rules apply.
matmul_fusion_rule = None
# Cast promoted BF16 vector-operation results back to BF16.
bf16_vector_keep_promoted = False

# Explicit kernel fallback policy.
force_fallback_kernel_names = {}

# Initialization and operator fallback policy.
online_acc_comp = os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY", "0") == "1"
# Comma-separated operator names to fall back, including GENERATE_LIST entries.
enable_full_lowering_fallback = os.environ.get("NPU_INDUCTOR_FALLBACK_LIST", "")

GENERATE_LIST = [
    control_deps,
    aten._assert_scalar,
    aten.mul,
    aten.add,
    aten.sub,
    aten.div,
    aten.clamp_min,
    aten.clamp_max,
    aten.maximum,
    aten.minimum,
    aten.abs,
    aten.reciprocal,
    aten.log,
    aten.exp,
    aten.pow,
    aten.sqrt,
    aten.rsqrt,
    aten.neg,
    aten.lt,
    aten.le,
    aten.gt,
    aten.ge,
    aten.eq,
    aten.ne,
    aten.bitwise_and,
    aten.bitwise_or,
    aten.bitwise_not,
    aten.where,
    prims.convert_element_type,
    torch.ops.npu.npu_dtype_cast,
    torch.ops.npu.npu_dtype_cast_backward,
    torch.ops.npu._npu_dtype_cast,
    torch.ops.npu._npu_dtype_cast_backward,
    aten.expand,
    aten.var_mean,
    aten.sum,
    aten.mean,
    aten.amax,
    aten.amin,
    aten.full,
    aten.relu,
    aten.where,
    aten.scalar_tensor,
    aten.unsqueeze,
    aten.squeeze,
    aten.reshape,
    aten.permute,
    torch.ops.higher_order.while_loop,
    torch.ops.higher_order.while_loop_stack_output,
    aten.copy,
    aten.copy_,
    aten.lift_fresh_copy,
    aten.lift_fresh_copy.default,
    aten.cat,
    aten.clone,
    aten.mm,
    aten.bmm,
    aten.addmm,
    aten.baddbmm,
    triton_kernel_wrapper_mutation,
    torch.ops.higher_order.flex_attention,
    torch.ops.higher_order.flex_attention_backward,
]

decomps_to_exclude_npu = [
    aten.gelu.default,
    aten.gelu_backward.default,
    aten.embedding,
    aten.embedding_backward,
    aten.embedding_dense_backward,
    aten.upsample_nearest2d,
    aten.upsample_nearest2d_backward,
    aten.upsample_nearest1d,
    aten.upsample_nearest1d_backward,
    aten.upsample_nearest3d,
    aten.upsample_nearest3d_backward,
    aten.upsample_bilinear2d,
    aten.upsample_bilinear2d_backward,
    aten.nll_loss2d_forward,
    aten.nll_loss2d_backward,
    aten.nll_loss_backward,
    aten.nll_loss_forward,
    aten.triu,
    aten.convolution_backward,
    aten._softmax_backward_data.default,
    aten.max_pool2d_with_indices,
    aten.max_pool2d_with_indices_backward,
    aten.slice.Tensor,
    aten.reflection_pad2d_backward,
    aten.reflection_pad2d,
    aten.grid_sampler_2d,
    aten.grid_sampler_2d_backward,
    aten.expm1,
]
