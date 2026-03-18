
import torch
from torch.distributed.tensor._op_schema import OpSchema, RuntimeSchemaInfo
from torch.distributed.tensor._ops.utils import register_op_strategy
from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy


npu = torch.ops.npu


custom_pointwise_ops = {
    npu.npu_dtype_cast.default: 0,
    npu._npu_dtype_cast.default: 0,
    npu.npu_dtype_cast_backward.default: 0,
    npu._npu_dtype_cast_backward.default: 0,
}


def custom_pointwise_strategy(op_schema: OpSchema):
    op_type = custom_pointwise_ops.get(op_schema.op, -1)
    return pointwise_strategy(op_schema, linearity=op_type)


for op in custom_pointwise_ops:
    register_op_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(custom_pointwise_strategy)
