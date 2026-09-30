"""Shared configuration for DVM Inductor integration."""

import os

# Run post-launch DVM debug checks.
debug_mode = False
# Emit standalone FX regression cases for DVM-fused graphs.
dump_fx_test = False
# Enable DVM cat fusion only when explicitly requested.
enable_cat = False
# Configure operator packets/overloads before the DVM backend is loaded.
# Disabling takes precedence when an operator occurs in both lists.
disable_decomp_list = []
enable_decomp_list = []
# C310 VF fusion: 0 off, 1 validates shapes, 2 assumes equal shapes.
vf_fusion = 0
# View-load fusion: 0 off, 1 requires a unit trailing stride,
# 2 allows arbitrary strides for vector kernels; mix uses 0, spec/split/concat at most 1.
view_fusion_level = 1
# Use DVM-specific fusion rules that prevent post-reduction fusion.
disable_post_reduce_fusion = False
# Enable DVM matmul fusion for mm, bmm, addmm, and baddbmm.
enable_matmul_fusion = (
    os.environ.get("INDUCTOR_DVM_ENABLE_MATMUL_FUSION", "0") == "1"
)
# Optional callback (n, m, k) for (..., m, k) @ (..., k, n).
# Returning False vetoes fusion; otherwise the existing fusion rules apply.
matmul_fusion_rule = None
# Cast promoted BF16 vector-operation results back to BF16.
bf16_vector_keep_promoted = False
