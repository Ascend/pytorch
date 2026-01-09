import functools
import logging
from typing import Any, Dict, List, Optional

import torch
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm
from torch._inductor.autoheuristic.autoheuristic_utils import (
    mm_operations,
)
import torch._inductor.codegen
from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate
import torch._inductor.kernel
from torch._inductor.virtualized import V

from torch._inductor import config as inductor_config, ir
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.ir import FixedLayout, FlexibleLayout, is_triton
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    NoValidChoicesError,
    TritonTemplate,
)
from torch._inductor.utils import (
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_cpp_gemm_template,
    use_max_autotune,
    is_pointwise_use,
)
from torch._inductor.kernel.mm_common import (
    _is_static_problem,
    addmm_epilogue,
    extra_mm_configs,
    mm_args,
    mm_configs,
    mm_grid,
    mm_options,
    triton_config,
)

from ..codegen.catlass.gemm_template import CATLASS1xGemmTemplate
from ..utils import use_catlass_template, use_triton_template


log = logging.getLogger("torch._inductor")
aten = torch.ops.aten

lazy_register_extern_choice = torch._inductor.kernel.mm.lazy_register_extern_choice
aten_mm = torch._inductor.kernel.mm.aten_mm
aten_addmm = torch._inductor.kernel.mm.aten_addmm
mm_config_kwargs = torch._inductor.kernel.mm.mm_config_kwargs
mm_autoheuristic = torch._inductor.kernel.mm.mm_autoheuristic
mm_template = torch._inductor.kernel.mm.mm_template


def is_contiguous_striding(size, stride) -> bool:
    def is_contiguous_row_major(stride, size) -> bool:
        # to support non-contiguous row-major input
        return V.graph.sizevars.statically_known_equals(stride[1], 1) and \
               V.graph.sizevars.statically_known_equals(stride[0], size[1])

    def is_contiguous_col_major(stride, size) -> bool:
        # to support non-contiguous col-major input
        return V.graph.sizevars.statically_known_equals(stride[0], 1) and \
               V.graph.sizevars.statically_known_equals(stride[1], size[0])

    return (
        is_contiguous_row_major(stride, size)
        or is_contiguous_col_major(stride, size)
    )


def _register_npu_inductor_mm():
    @register_lowering(aten.mm, type_promotion_kind=None)
    def tuned_mm(mat1, mat2, *, layout=None):
        m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
        name = "mm"

        aten_layout = layout
        if not use_max_autotune():
            aten_layout = FlexibleLayout(
                device=layout.device, dtype=layout.dtype, size=layout.size
            )

        # options to tune from
        choices = (
            [aten_mm.bind((mat1, mat2), aten_layout)] if use_aten_gemm_kernels() else []
        )
        _, is_nonzero = _is_static_problem(layout)

        is_contiguous_input = (
            is_contiguous_striding(mat1.get_size(), mat1.get_stride())
            and is_contiguous_striding(mat2.get_size(), mat2.get_stride())
        )
        if (
            is_contiguous_input
            and is_nonzero
            and use_catlass_template("mm", layout, m, n, k)
        ):
            CATLASS1xGemmTemplate.add_catlass_gemm_choices(
                choices, layout, [mat1, mat2]
            )
            # debug log
            log.info(f"Choices number after catlass template: {len(choices)}")

        if is_nonzero and use_ck_gemm_template(layout, m, n, k):
            CKGemmTemplate.add_ck_gemm_choices(choices, layout, [mat1, mat2])

        if use_cpp_gemm_template(layout, mat1, mat2):
            CppGemmTemplate.add_choices(
                choices,
                layout,
                [mat1, mat2],
            )

        input_nodes = [mat1, mat2]

        if (
            len(choices) == 0
            and not use_aten_gemm_kernels()
            and inductor_config.autotune_fallback_to_aten
        ):
            log.warning("No choices for GEMM, using ATen backend as fallback")
            return aten_mm.bind((mat1, mat2), aten_layout).output_node()

        for k in inductor_config.external_matmul:
            choices.append(lazy_register_extern_choice(k).bind((mat1, mat2), layout))

        try:
            return autotune_select_algorithm(name, choices, [mat1, mat2], layout)
        except NoValidChoicesError:
            if not inductor_config.autotune_fallback_to_aten:
                raise
            log.warning(
                "All choices for GEMM were invalid, using ATen backend as fallback"
            )
            return aten_mm.bind((mat1, mat2), aten_layout).output_node()


def _register_npu_inductor_addmm():
    @register_lowering(aten.addmm, type_promotion_kind=None)
    def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
        ordered_kwargs_for_cpp_kernel = ("beta", "alpha")
        m, n, k, layout, mat1, mat2, inp_expanded = mm_args(
            mat1, mat2, inp, layout=layout
        )
        static_shape, is_nonzero = _is_static_problem(layout)
        if (not is_nonzero) or (not use_max_autotune()):
            if isinstance(layout, FixedLayout):
                layout = FlexibleLayout(
                    device=layout.device, dtype=layout.dtype, size=layout.size
                )
            choices = (
                [
                    aten_addmm.bind(
                        (inp, mat1, mat2),
                        layout,
                        alpha=alpha,
                        beta=beta,
                    )
                ]
                if use_aten_gemm_kernels()
                else []
            )
            return autotune_select_algorithm(
                "addmm", choices, [inp, mat1, mat2], layout
            )

        choices = (
            [
                aten_addmm.bind(
                    (inp_expanded, mat1, mat2),
                    layout,
                    alpha=alpha,
                    beta=beta,
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )

        is_contiguous_input = (
            is_contiguous_striding(mat1.get_size(), mat1.get_stride())
            and is_contiguous_striding(mat2.get_size(), mat2.get_stride())
        )
        if (
            is_contiguous_input
            and static_shape
            and is_nonzero
            and use_catlass_template("addmm", layout, m, n, k)
        ):
            # Filter out broadcasting on the last dim of the bias term
            # since catlass does not support it yet.
            if (
                PythonWrapperCodegen.statically_known_int_or_none(
                    inp_expanded.layout.stride[-1]
                )
                != 0
            ):
                CATLASS1xGemmTemplate.add_catlass_gemm_choices(
                    choices,
                    layout,
                    [mat1, mat2, inp_expanded],
                    alpha=alpha,
                    beta=beta,
                )

        if is_nonzero and use_ck_gemm_template(layout, m, n, k):
            CKGemmTemplate.add_ck_gemm_choices(
                choices,
                layout,
                [mat1, mat2, inp_expanded],
                alpha=alpha,
                beta=beta,
            )

        if use_cpp_gemm_template(layout, mat1, mat2):
            CppGemmTemplate.add_choices(
                choices,
                layout,
                [inp_expanded, mat1, mat2],
                alpha=alpha,
                beta=beta,
                has_bias=True,
            )

        add_aten_fallback = False
        if len(choices) == 0:
            log.warning("No choices for GEMM, using ATen backend as fallback")
            add_aten_fallback = True

        if add_aten_fallback:
            choices.append(
                aten_addmm.bind(
                    (inp_expanded, mat1, mat2),
                    layout,
                    ordered_kwargs_for_cpp_kernel,
                    alpha=alpha,
                    beta=beta,
                )
            )

        try:
            return autotune_select_algorithm(
                "addmm", choices, [inp_expanded, mat1, mat2], layout
            )
        except NoValidChoicesError:
            if not inductor_config.autotune_fallback_to_aten:
                raise
            log.warning(
                "All choices for GEMM were invalid, using ATen backend as fallback"
            )
            fallback_choice = aten_addmm.bind(
                (inp, mat1, mat2),
                layout,
                ordered_kwargs_for_cpp_kernel,
                alpha=alpha,
                beta=beta,
            )
            return fallback_choice.output_node()
