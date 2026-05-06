import torch
from torch.distributed.tensor._op_schema import OpSchema, RuntimeSchemaInfo
from torch_npu._compat.distributed import (
    register_op_strategy,
    pointwise_strategy,
    register_single_dim_strategy,
    _ShardingPlaceholder,
)
from torch_npu._compat.version import CURRENT_VERSION

npu = torch.ops.npu

custom_pointwise_ops = {
    npu.npu_dtype_cast.default: 0,
    npu._npu_dtype_cast.default: 0,
    npu.npu_dtype_cast_backward.default: 0,
    npu._npu_dtype_cast_backward.default: 0,
}


def _custom_pointwise_strategy_new(
    op: torch._ops.OpOverload,
    args_schema,
    kwargs_schema,
):
    """
    New strategy function for PyTorch 2.11+ using register_single_dim_strategy.
    Returns placements list following the new API format: list[list[Placement | _ShardingPlaceholder]]

    For pointwise ops, all tensors should shard on the same dimension.
    This matches the behavior of pointwise_strategy(linearity=0) in older PyTorch.
    """
    from torch.distributed.tensor.placement_types import Replicate
    from torch.distributed.tensor._dtensor_spec import TensorMeta

    # Get tensor arguments
    tensor_args = [arg for arg in args_schema if isinstance(arg, TensorMeta)]
    if not tensor_args:
        # No tensor args, return replicate strategy
        return [[Replicate()]]

    # Get common shape (broadcasted)
    common_shape = torch.broadcast_shapes(*[arg.shape for arg in tensor_args])

    num_outputs = 1  # Most pointwise ops have single output
    placements = []

    # For each dimension, create a sharding strategy
    for dim_idx in range(len(common_shape)):
        strategy = [_ShardingPlaceholder(dim_idx)] * num_outputs
        for arg in tensor_args:
            # Map common dim to arg dim (handling broadcast)
            common_dim_to_arg_dim = _infer_broadcast_dims_map(common_shape, arg.shape)
            if common_dim_to_arg_dim[dim_idx] >= 0:
                strategy.append(_ShardingPlaceholder(common_dim_to_arg_dim[dim_idx]))
            else:
                strategy.append(Replicate())
        placements.append(strategy)

    # Add replicate strategy
    replicate_strategy = [Replicate()] * (num_outputs + len(tensor_args))
    placements.insert(0, replicate_strategy)

    return placements


def _infer_broadcast_dims_map(common_shape, arg_shape):
    """Map dimensions from common shape to argument shape for broadcast handling."""
    # Simple implementation: map matching dimensions, -1 for broadcast dimensions
    result = []
    arg_ndim = len(arg_shape)
    common_ndim = len(common_shape)

    for common_idx in range(common_ndim):
        # Calculate corresponding arg index
        arg_idx = common_idx - (common_ndim - arg_ndim)
        if arg_idx >= 0 and arg_idx < arg_ndim:
            # Check if dimension matches (not broadcast)
            if arg_shape[arg_idx] == common_shape[common_idx] or arg_shape[arg_idx] == 1:
                result.append(arg_idx)
            else:
                result.append(-1)
        else:
            result.append(-1)

    return result


def custom_pointwise_strategy(op_schema: OpSchema):
    """Legacy strategy function for PyTorch < 2.11."""
    op_type = custom_pointwise_ops.get(op_schema.op, -1)
    return pointwise_strategy(op_schema, linearity=op_type)


# Register strategies based on PyTorch version
if CURRENT_VERSION >= (2, 11):
    # Use new register_single_dim_strategy API for PyTorch 2.11+
    if register_single_dim_strategy is not None:
        for op in custom_pointwise_ops:
            register_single_dim_strategy(
                op,
                schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
            )(_custom_pointwise_strategy_new)
    else:
        # Fallback if register_single_dim_strategy is not available
        # This should not happen in PyTorch 2.11+, but we keep it for safety
        pass
else:
    # Use legacy pointwise_strategy for PyTorch < 2.11
    if pointwise_strategy is not None:
        for op in custom_pointwise_ops:
            register_op_strategy(
                op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
            )(custom_pointwise_strategy)
    else:
        # Fallback if pointwise_strategy is not available
        pass