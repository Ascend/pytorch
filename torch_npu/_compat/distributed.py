from torch_npu._compat.version import CURRENT_VERSION

# COMPAT(>= 2.11): register_op_strategy / register_prop_rule moved from
#   _ops.registration to _ops.utils in PyTorch 2.11.
# CAN REMOVE else branch when MIN_SUPPORTED >= (2, 11)
if CURRENT_VERSION >= (2, 11):
    from torch.distributed.tensor._ops.utils import register_op_strategy, register_prop_rule
else:
    from torch.distributed.tensor._ops.registration import register_op_strategy, register_prop_rule

# COMPAT(>= 2.11): pointwise_strategy removed in PyTorch 2.11
#   New API uses register_single_dim_strategy and _register_single_dim_pointwise
# CAN REMOVE else branch when MIN_SUPPORTED >= (2, 11)
if CURRENT_VERSION >= (2, 11):
    try:
        from torch.distributed.tensor._ops.single_dim_strategy import (
            register_single_dim_strategy,
            _ShardingPlaceholder,
        )
        # pointwise_strategy is no longer available, use register_single_dim_strategy instead
        pointwise_strategy = None
    except ImportError:
        # Fallback for older PyTorch versions without single_dim_strategy
        pointwise_strategy = None
        register_single_dim_strategy = None
        _ShardingPlaceholder = None
else:
    try:
        from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy
        register_single_dim_strategy = None
        _ShardingPlaceholder = None
    except ImportError:
        pointwise_strategy = None
        register_single_dim_strategy = None
        _ShardingPlaceholder = None
