import logging
from dataclasses import dataclass
from typing import Any, List, Union, Optional

import torch
from torch._dynamo.utils import counters
from torch._inductor.runtime.triton_compat import tl
from torch._inductor.virtualized import V
from torch._inductor.lowering import fallback_handler
from torch.utils._triton import has_triton

from torch._inductor.ir import Layout, TensorBox
from torch._inductor.lowering import register_lowering
from torch._inductor.kernel.mm_common import _is_static_problem
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    TritonTemplate,
)
from torch._inductor.utils import (
    use_aten_gemm_kernels,
    sympy_product,
)

from .mm import is_contiguous_striding

from ..utils import use_catlass_template
from ..codegen.catlass.gemm_template import CATLASS1xGemmTemplate


log = logging.getLogger(__name__)
aten = torch.ops.aten

TORCH_DTYPE_MAP = {
    torch.float16: 5,
    torch.bfloat16: 15,
    torch.float32: 6,
    torch.float8_e5m2: 23,
    torch.float8_e4m3fn: 24,
    torch.bits8: 21,
    torch.int8: 1,
    torch.int32: 3,
}

TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    1: torch.int8,
    3: torch.int32,
    5: torch.float16,
    6: torch.float32,
    15: torch.bfloat16,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
}


def is_batch_stride_largest_or_zero(mat) -> bool:
    """
    Checking if the batch stride is the largest in the stride.
    """
    size = mat.get_size()
    stride = mat.get_stride()
    assert len(size) == len(stride) == 3, "Expect 3D tensors"
    if stride[0] != 0 and stride[0] != sympy_product(size[1:]):
        return False

    return True


def grouped_mm_args(
    mat1: List[TensorBox],
    mat2: List[TensorBox],
    offs: Optional[TensorBox],
):
    mat1, mat2 = realize_inputs(mat1[0], mat2[0])
    if offs is not None:
        realize_inputs(offs)
    mat1_size = mat1.get_size()
    mat2_size = mat2.get_size()

    from torch._inductor.ir import FixedLayout

    out_dtype = mat1.get_dtype()
    out_size = [mat1_size[0], mat2_size[-1]]
    out_stride = [out_size[-1], 1]

    layout = FixedLayout(
        mat1.get_device(),
        out_dtype,
        out_size,
        out_stride,
    )

    return (mat1_size, mat2_size, layout, mat1, mat2, offs)


def create_offsets(x, m1_size, m2_size, offs_size):
    m = V.graph.sizevars.size_hint(m1_size[0])
    noffs = V.graph.sizevars.size_hint(offs_size[0])
    step = m / noffs

    # Note: NPU cannot support linspace with int64, in case x.dtype = torch.int64, we implemented as following
    result = torch.linspace(
        step, m, noffs, dtype=torch.int32, device=x.get_device()
    )
    return result.to(x.get_dtype())


def check_catlass_support(
    mat_a: List[TensorBox],
    mat_b: List[TensorBox],
    bias: Optional[List[TensorBox]] = None,
    scale: Optional[List[TensorBox]] = None,
    offset: Optional[List[TensorBox]] = None,
    group_list: Optional[Union[List[int], TensorBox]] = None,
    group_type: int = None,
    group_list_type: int = None,
    act_type: int = None,
    output_dtype: Optional[torch.dtype] = None,
) -> bool:
    # Catlass only support single X and single W input
    if len(mat_a) != 1 or len(mat_b) != 1:
        return False

    if len(mat_a[0].get_size()) != 2 or len(mat_b[0].get_size()) != 3:
        return False
    
    # Catlass currently not support group mm with bias
    if bias:
        return False

    if scale is not None and scale != 1:
        return False

    if offset:
        return False 

    if group_list is None or not isinstance(group_list, TensorBox):
        return False

    if group_list.get_size()[0] != mat_b[0].get_size()[0]:
        return False
    
    # Catlass only support splitting in m axis
    if group_type is not None and group_type != 0:
        return False

    # Catlass only support group list with cumsum result
    if group_list_type is not None and group_list_type != 0:
        return False

    if act_type is not None and act_type != 0:
        return False

    if mat_a[0].get_dtype() != mat_b[0].get_dtype():
        return False

    if output_dtype is not None and (
        output_dtype not in TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP or
        TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[output_dtype] != mat_a[0].get_dtype()
    ):
        return False

    return True


def _tuned_grouped_mm_common(
    operator_name: str,
    algorithm_name: str,
    extern_kernel_choice: ExternKernelChoice,
    mat_a: List[TensorBox],
    mat_b: List[TensorBox],
    bias: Optional[List[TensorBox]] = None,
    scale: Optional[List[TensorBox]] = None,
    offset: Optional[List[TensorBox]] = None,
    group_list: Optional[Union[List[int], TensorBox]] = None,
    group_type: int = None,
    group_list_type: int = None,
    act_type: int = None,
    output_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> List[TensorBox]:
    catlass_compatible = check_catlass_support(mat_a, mat_b, bias, scale, offset, 
        group_list, group_type, group_list_type, act_type, output_dtype
    )
    if not catlass_compatible:
        return fallback_handler(torch.ops.npu.npu_grouped_matmul.default)(
            mat_a,
            mat_b,
            bias=bias,
            scale=scale,
            offset=offset,
            group_list=group_list,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            output_dtype=output_dtype,
            **kwargs,
        )


    m1_size, m2_size, layout, mat_a, mat_b, offs = grouped_mm_args(
        mat_a, mat_b, group_list
    )

    counters["aten_mm_info"][operator_name] += 1
    log_message = f"Tuned {operator_name}: mat1_shape=%s, mat2_shape=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s"
    log.info(
        log_message,
        m1_size,
        m2_size,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    # workaround for Inductor not supporting optional tensor input arguments
    input_nodes: list[Any] = [mat_a, mat_b]
    if offs is not None:
        input_nodes.append(realize_inputs(offs))

    choices = []

    _, is_nonzero = _is_static_problem(layout)

    m, n, k = mat_a.get_size()[0], mat_b.get_size()[-1], mat_a.get_size()[1]

    batch_stride_largest_or_zero = is_batch_stride_largest_or_zero(mat_b)
    is_contiguous_input = False
    if batch_stride_largest_or_zero:
        is_contiguous_input = (
            is_contiguous_striding(mat_a.get_size(), mat_a.get_stride())
            and is_contiguous_striding(mat_b.get_size()[1:], mat_b.get_stride()[1:])
        )

    if is_contiguous_input and is_nonzero and use_catlass_template("grouped_mm", layout, m, n, k):
        CATLASS1xGemmTemplate.add_catlass_gemm_choices(
            choices, 
            layout, 
            [mat_a, mat_b, bias, offs], # currently catlass does not support grouped mm with bias
        )
        # debug log
        log.info(f"Choices number after catlass template: {len(choices)}")
    else:
        return fallback_handler(torch.ops.npu.npu_grouped_matmul.default)(
            [mat_a],
            [mat_b],
            bias=bias,
            scale=scale,
            offset=offset,
            group_list=group_list,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            output_dtype=output_dtype,
            **kwargs,
        )

    if not choices:
        return fallback_handler(torch.ops.npu.npu_grouped_matmul.default)(
            [mat_a],
            [mat_b],
            bias=bias,
            scale=scale,
            offset=offset,
            group_list=group_list,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            output_dtype=output_dtype,
            **kwargs,
        )

    input_gen_fns = {
        2: lambda x: create_offsets(
            x, m1_size, m2_size, offs.get_size()
        ),
    }

    tb = autotune_select_algorithm(
        algorithm_name, choices, input_nodes, layout, input_gen_fns=input_gen_fns
    )
    return [tb]


def _register_npu_inductor_grouped_mm():
    @register_lowering(torch.ops.npu.npu_grouped_matmul, type_promotion_kind=None)
    def tuned_grouped_mm(
        mat_a: List[TensorBox],
        mat_b: List[TensorBox],
        bias: Optional[List[TensorBox]] = None,
        scale: Optional[List[TensorBox]] = None,
        offset: Optional[List[TensorBox]] = None,
        group_list: Optional[Union[List[int], TensorBox]] = None,
        group_type: int = None,
        group_list_type: int = None,
        act_type: int = None,
        output_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> List[TensorBox]:
        """Auto-tuning for _grouped_mm() operator."""
        return _tuned_grouped_mm_common(
            operator_name="torch.ops.npu.npu_grouped_matmul",
            algorithm_name="grouped_mm",
            extern_kernel_choice=None, # aten__grouped_mm,
            mat_a=mat_a,
            mat_b=mat_b,
            bias=bias,
            scale=scale,
            offset=offset,
            group_list=group_list,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            output_dtype=output_dtype,
            **kwargs,
        )
