import functools
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate

from torch._inductor import ir, lowering as L
from torch._inductor.lowering import fallback_handler
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    SymbolicGridFn,
    TritonTemplate,
)
from torch._inductor.utils import (
    ceildiv as cdiv,
    use_aten_gemm_kernels,
    use_ck_template,
    use_cpp_bmm_template,
    sympy_product,
)
from torch._inductor.virtualized import V
from torch._inductor.kernel.mm_common import (
    _is_static_problem,
    addmm_epilogue,
    mm_args,
)
from torch._inductor.kernel import bmm as inductor_bmm

from .mm import is_contiguous_striding
from ..select_algorithm import NPUTritonTemplate
from ..utils import use_catlass_template, use_triton_template


log = logging.getLogger("torch._inductor")
aten = torch.ops.aten

aten_bmm = inductor_bmm.aten_bmm
aten_baddbmm = inductor_bmm.aten_baddbmm


# ---------------------------------------------------------------------------
# NPU Triton BMM Template (for CV / epilogue fusion with batch dimension)
# ---------------------------------------------------------------------------
# Uses triton_bmm.py.jinja which extends the mm template with a batch
# dimension (idx_q = tl.program_id(1)).  Grid is (MN_tiles, batch, 1).
# The {{store_output}} placeholder supports epilogue fusion (e.g. relu).

@SymbolicGridFn
def npu_bmm_grid(b, m, n, meta, *, cdiv):
    """Grid function for NPU bmm triton template.

    Returns (num_mn_tiles, batch, 1) where num_mn_tiles covers all M*N blocks.
    """
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), b, 1)


# Inline template source (previously loaded from templates/triton_bmm.py.jinja).
# Kept as a string constant so the kernel no longer depends on the external
# .jinja file at runtime.
_BMM_TEMPLATE = """{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # batch dimension index — precompute batch offsets before K-loop
    # to avoid redundant multiply inside the hot loop
    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    a_batch_off = idx_q * stride_aq
    b_batch_off = idx_q * stride_bq

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        {% if EVEN_K %}
        a = tl.load(A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak + a_batch_off))
        b = tl.load(B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn + b_batch_off))
        {% else %}
        # K is not a multiple of BLOCK_K: mask out-of-bounds elements
        k_mask = offs_k < K
        a = tl.load(A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak + a_batch_off), mask=k_mask[None, :], other=0.0)
        b = tl.load(B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn + b_batch_off), mask=k_mask[:, None], other=0.0)
        {% endif %}
        acc = tl.dot(a, b, acc=acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm, rn and idx_q to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
"""

npu_triton_bmm_template = NPUTritonTemplate(
    name="npu_triton_bmm",
    grid=npu_bmm_grid,
    source=_BMM_TEMPLATE,
    debug=False,
)


def _get_npu_bmm_configs(
    m: int,
    n: int,
    k: int,
) -> List[Dict[str, Any]]:
    """Generate tiling configs for NPU triton bmm template.

    Same tiling shapes as mm, adapted for batched matmul.
    """
    configs: List[Dict[str, Any]] = []

    tile_shapes = [
        (64, 64, 32),
        (64, 128, 32),
        (128, 64, 32),
        (128, 128, 32),
        (64, 64, 64),
        (64, 64, 128),
        (64, 64, 256),
        (64, 256, 256),
        (128, 64, 64),
        (64, 128, 64),
        (128, 128, 64),    # large tile for big BMM shapes
        # --- added by performance optimization (autotune-discovered) ---
        # These configs were found to be significantly faster on ascend950PR
        # for large BMM shapes (e.g. B=80, M=200, N=1280, K=640).
        # BLOCK_N=256 improves N-dimension tiling for wide output matrices.
        (128, 256, 64),
        (64, 256, 64),
        # BLOCK_K=128 reduces K-loop iterations for large K dimensions.
        (128, 128, 128),
        (64, 128, 128),
        # BLOCK_K=256 further reduces K-loop iterations (e.g. K=1280 → 5 iters
        # instead of 10 with BLOCK_K=128). Best config for large K on ascend950PR.
        (128, 256, 256),
        (128, 128, 256),
        # --- end of performance optimization additions ---
        (32, 64, 32),
        (64, 32, 32),
        (32, 32, 32),
        (32, 32, 128)
    ]

    for block_m, block_n, block_k in tile_shapes:
        # Dynamically compute EVEN_K: True only when K is an exact multiple
        # of BLOCK_K, so the template can skip the K-boundary mask
        # for performance while staying correct when K is not aligned.
        even_k = (k % block_k == 0)
        # Use GROUP_M=[1, 8] for large tiles (BLOCK_M>=128 and BLOCK_N>=128),
        # GROUP_M=[8] for small tiles. GROUP_M=1 (row-major traversal) is
        # better for shapes with small grid_m (e.g. M=200, BLOCK_M=128 → grid_m=2).
        if block_m >= 128 and block_n >= 128:
            group_m_values = [1, 8]
        else:
            group_m_values = [8]
        for group_m in group_m_values:
            for num_stages in [2, 3]:
                configs.append({
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": group_m,
                    "num_stages": num_stages,
                    "num_warps": 4,
                    "ALLOW_TF32": "False",
                    "ACC_TYPE": "tl.float32",
                    "EVEN_K": even_k,
                })

    return configs


def add_npu_triton_bmm_choices(
    choices: List[ir.ChoiceCaller],
    layout: "ir.Layout",
    mat1: "ir.IRNode",
    mat2: "ir.IRNode",
    m: int,
    n: int,
    k: int,
) -> None:
    """Add NPU Triton bmm template choices to the choices list.

    The bmm template handles the batch dimension via tl.program_id(1) and
    supports epilogue fusion via {{store_output}}.
    """
    input_nodes = [mat1, mat2]
    configs = _get_npu_bmm_configs(m, n, k)

    for cfg in configs:
        num_stages = cfg.pop("num_stages")
        num_warps = cfg.pop("num_warps")

        try:
            choice = npu_triton_bmm_template.generate(
                input_nodes=input_nodes,
                layout=layout,
                num_stages=num_stages,
                num_warps=num_warps,
                **cfg,
            )
            if choice is not None:
                choices.append(choice)
        except Exception as e:
            log.debug(
                "Failed to generate NPU triton bmm choice with config %s: %s",
                cfg,
                e,
            )


def is_batch_stride_largest_or_zero(mat1, mat2, layout) -> bool:
    """
    Checking if the batch stride is the largest in the stride.
    """
    sizes = [mat1.get_size(), mat2.get_size(), layout.size]
    strides = [mat1.get_stride(), mat2.get_stride(), layout.stride]
    for size, stride in zip(sizes, strides):
        assert len(size) == len(stride) == 3, "Expect 3D tensors"
        if stride[0] != 0 and stride[0] != sympy_product(size[1:]):
            return False

    return True


def _register_npu_inductor_bmm():
    @L.register_lowering(aten.bmm)
    def tuned_bmm(mat1, mat2, *, layout=None):
        if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
            # decompose to small ops when memory bound
            if mat1.get_size()[1] == 1 or mat2.get_size()[2] == 1:
                mat1 = L.unsqueeze(mat1, -1)
                mat2 = L.unsqueeze(mat2, 1)
                return L.sum_(L.mul(mat1, mat2), axis=2)

            def is_valid_to_require_contiguous(t):
                if not ir.is_storage_and_layout(t):
                    return True
                _, layout = ir.as_storage_and_layout(t, freeze=False)
                return isinstance(layout, ir.FlexibleLayout)

            def is_preferred_layout_as_bmm_input(sizes, strides):
                # contiguous on one of the last two dims
                return (
                    strides[-1] == 1 and (sizes[-2] == 1 or strides[-2] >= sizes[-1])
                ) or (strides[-2] == 1 and (sizes[-1] == 1 or strides[-1] >= sizes[-2]))

            # Make the input of bmm contiguous
            # if it is not contiguous on either of the last two dims,
            # because bmm cpu implementation would do contiguous() if not.
            # This is to avoid additional copies in bmm.
            def may_require_contiguous(t, meta_t):
                sizes = meta_t.meta["val"].size()
                strides = meta_t.meta["val"].stride()
                if not is_preferred_layout_as_bmm_input(sizes, strides):
                    t = ir.ExternKernel.require_contiguous(t)
                return t

            if is_valid_to_require_contiguous(mat1):
                meta_mat1 = V.graph.current_node.args[0]
                mat1 = may_require_contiguous(mat1, meta_mat1)
            if is_valid_to_require_contiguous(mat2):
                meta_mat2 = V.graph.current_node.args[1]
                mat2 = may_require_contiguous(mat2, meta_mat2)

        m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

        # options to tune from
        choices = (
            [aten_bmm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
        )
        static_shape, is_nonzero = _is_static_problem(layout)
        batch_stride_largest_or_zero = is_batch_stride_largest_or_zero(mat1, mat2, layout)
        is_contiguous_input = False
        if batch_stride_largest_or_zero:
            is_contiguous_input = (
                is_contiguous_striding(mat1.get_size()[1:], mat1.get_stride()[1:])
                and is_contiguous_striding(mat2.get_size()[1:], mat2.get_stride()[1:])
            )
        if (
            is_contiguous_input
            and static_shape
            and is_nonzero
            and use_catlass_template("bmm", layout, m, n, k)
        ):
            from ..codegen.catlass.gemm_template import CATLASS1xGemmTemplate

            CATLASS1xGemmTemplate.add_catlass_gemm_choices(
                choices, layout, [mat1, mat2]
            )

        if use_cpp_bmm_template(layout, mat1, mat2):
            from torch._inductor.codegen.cpp_bmm_template import CppBmmTemplate

            CppBmmTemplate.add_choices(
                choices,
                layout,
                [mat1, mat2],
            )
        if use_ck_template(layout):
            CKGemmTemplate.add_ck_gemm_choices(choices, layout, [mat1, mat2])

        # Add NPU Triton bmm template choices for CV (Compute/Vector) fusion.
        # The bmm template handles the batch dimension via tl.program_id(1)
        # and supports epilogue fusion via {{store_output}}.
        if is_nonzero and use_triton_template(layout):
            try:
                add_npu_triton_bmm_choices(
                    choices, layout, mat1, mat2, m, n, k
                )
                log.debug(
                    "NPU Triton CV fusion: added triton bmm template choices "
                    "for bmm(%d, %d, %d), total choices now %d",
                    m,
                    n,
                    k,
                    len(choices),
                )
            except Exception as e:
                log.warning("Failed to add NPU triton bmm template choices: %s", e)

        if len(choices) == 0:
            log.warning("No choices for GEMM, using ATen backend as fallback")
            choices.append(aten_bmm.bind((mat1, mat2), layout))

        return autotune_select_algorithm("bmm", choices, [mat1, mat2], layout)
