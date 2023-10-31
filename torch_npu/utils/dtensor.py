import torch
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule, normalize_dims
from torch.distributed._tensor.ops.matrix_ops import bmm_strategy
from torch.distributed._tensor.ops.view_ops import (
    register_prop_rule_map,
    view_groups,
    Op,
    ops,
    InputDim
)
import torch_npu

npu = torch.ops.npu
aten = torch.ops.aten

pointwise_ops = [
    # please keep the entries below alphabetically sorted
    # native ops
    aten.isclose.default,
    aten.isfinite.default,
    aten.true_divide.Scalar,
    aten.true_divide.Tensor,
    aten.true_divide.out,
    aten.true_divide_.Scalar,
    aten.true_divide_.Tensor,
    # custom ops
    npu.fast_gelu.default,
    npu.npu_dtype_cast.default,
    npu.npu_fast_gelu.default,
    npu.npu_layer_norm_eval.default,
    # backward point-wise ops
    # please keep the entries below alphabetically sorted
    npu.npu_fast_gelu_backward.default
]

matrix_ops = [
    npu.npu_bmmV2.default
]


def register_ops_under_dtensor_rules():
    # pointwise rule
    for op in pointwise_ops:
        register_prop_rule(op)(pointwise_rule)

    # bmm rules
    for op in matrix_ops:
        register_prop_rule(op)(bmm_strategy)

    # reshape_prop under view_ops
    ops[torch_npu.npu_transpose] = Op(
        dim_map=lambda input, dims: tuple(
            InputDim(i) for i in normalize_dims(dims, input.ndim)
        )
    )
    register_prop_rule_map(npu.npu_transpose.default, torch_npu.npu_transpose)
