import logging

import torch
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate

from torch._inductor import ir, lowering as L
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
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
    mm_configs,
    mm_options,
)

from .mm import is_contiguous_striding
from ..utils import use_catlass_template, use_triton_template


log = logging.getLogger("torch._inductor")
aten = torch.ops.aten

aten_bmm = torch._inductor.kernel.bmm.aten_bmm
aten_baddbmm = torch._inductor.kernel.bmm.aten_baddbmm
bmm_configs = torch._inductor.kernel.bmm.bmm_configs
bmm_template = torch._inductor.kernel.bmm.bmm_template


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

        if len(choices) == 0:
            log.warning("No choices for GEMM, using ATen backend as fallback")
            choices.append(aten_bmm.bind((mat1, mat2), layout))

        return autotune_select_algorithm("bmm", choices, [mat1, mat2], layout)
