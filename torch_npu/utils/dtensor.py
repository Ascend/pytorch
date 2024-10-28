import torch
from torch.distributed.tensor._ops._common_rules import pointwise_rule
from torch.distributed.tensor._ops.utils import register_prop_rule, normalize_dims
from torch.distributed.tensor._ops._matrix_ops import bmm_strategy
from torch.distributed.tensor._ops._view_ops import (
    register_op_strategy_map,
    dim_maps,
    InputDim
)
import torch_npu

__all__ = []


def _register_ops_under_dtensor_rules():
    npu = torch.ops.npu
    aten = torch.ops.aten

    pointwise_ops = [
        # please keep the entries below alphabetically sorted
        # native ops
        aten.isclose.default,
        aten.isfinite.default,
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
    # pointwise rule
    for op in pointwise_ops:
        register_prop_rule(op)(pointwise_rule)

    # bmm rules
    for op in matrix_ops:
        register_prop_rule(op)(bmm_strategy)

    # reshape_prop under view_ops
    dim_maps.update({
        torch_npu.npu_transpose: lambda input, dims: tuple(
            InputDim(i) for i in normalize_dims(dims, input.ndim)
        )
    })
    register_op_strategy_map(npu.npu_transpose.default, torch_npu.npu_transpose)
