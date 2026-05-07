
import torch
from torch.distributed.tensor._op_schema import OpSchema, RuntimeSchemaInfo
from torch_npu._compat.distributed import register_op_strategy
from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy

aten = torch.ops.aten
npu = torch.ops.npu


custom_linear_pointwise_ops = {
    npu.npu_dtype_cast.default: 0,
    npu._npu_dtype_cast.default: 0,
    npu.npu_dtype_cast_backward.default: 0,
    npu._npu_dtype_cast_backward.default: 0,
}


def custom_linear_pointwise_strategy(op_schema: OpSchema):
    op_type = custom_linear_pointwise_ops.get(op_schema.op, -1)
    return pointwise_strategy(op_schema, linearity=op_type)


for op in custom_linear_pointwise_ops:
    register_op_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(custom_linear_pointwise_strategy)


custom_pointwise_ops = [
    # please keep the entries below alphabetically sorted
    # native ops
    aten.isclose.default,
    aten.isfinite.default,
    # custom ops
    npu.fast_gelu.default,
    npu.npu_fast_gelu.default,
    npu.npu_layer_norm_eval.default,
    # backward point-wise ops
    # please keep the entries below alphabetically sorted
    npu.npu_fast_gelu_backward.default
]


for op in custom_pointwise_ops:
    register_op_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(pointwise_strategy)
