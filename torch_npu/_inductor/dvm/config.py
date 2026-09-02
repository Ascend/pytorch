"""Shared configuration for DVM Inductor integration."""

import os

# Run post-launch DVM debug checks.
debug_mode = False
# Emit standalone FX regression cases for DVM-fused graphs.
dump_fx_test = False
# View-load fusion: 0 off, 1 requires a unit trailing stride, 2 always on.
view_fusion_level = 1
# Use DVM-specific fusion rules that prevent post-reduction fusion.
disable_post_reduce_fusion = False
# Enable DVM matmul fusion for mm, bmm, addmm, and baddbmm.
enable_matmul_fusion = (
    os.environ.get("INDUCTOR_DVM_ENABLE_MATMUL_FUSION", "0") == "1"
)
# Cast promoted BF16 vector-operation results back to BF16.
bf16_vector_keep_promoted = False
