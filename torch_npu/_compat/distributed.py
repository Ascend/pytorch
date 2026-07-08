from torch_npu._compat.version import CURRENT_VERSION

__all__ = [
    "register_op_strategy",
    "register_prop_rule",
    "_mm_like_strategy",
]

# COMPAT(>= 2.11): register_op_strategy / register_prop_rule moved from
#   _ops.registration to _ops.utils in PyTorch 2.11.
# CAN REMOVE else branch when MIN_SUPPORTED >= (2, 11)
if CURRENT_VERSION >= (2, 11):
    from torch.distributed.tensor._ops.utils import register_op_strategy, register_prop_rule
else:
    from torch.distributed.tensor._ops.registration import register_op_strategy, register_prop_rule


# COMPAT(>= 2.14): upstream pytorch#186667 removed the helper
#   torch.distributed.tensor._ops._matrix_ops._mm_like_strategy as part of the
#   matrix_ops "single dim strategies" refactor. All three helpers the old
#   implementation depended on (gen_einsum_strategies, is_tensor_shardable,
#   generate_redistribute_costs) are still exported, so we replay the
#   pre-refactor body inline. This keeps torch_npu's custom_bmm_strategy
#   callsite (which still uses the old @register_op_strategy pipeline)
#   untouched, and does not require migrating to the new
#   register_single_dim_strategy interface.
# CAN REMOVE else branch when MIN_SUPPORTED >= (2, 14)
if CURRENT_VERSION >= (2, 14):
    from torch.distributed.tensor._op_schema import OpStrategy
    from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
    from torch.distributed.tensor._ops.utils import (
        is_tensor_shardable,
        generate_redistribute_costs,
    )

    def _mm_like_strategy(mm_equation, mesh, op_schema):
        self_strategy, mat2_strategy = op_schema.args_schema
        if not isinstance(self_strategy, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {type(self_strategy)}")
        if not isinstance(mat2_strategy, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {type(mat2_strategy)}")
        mm_strategy = gen_einsum_strategies(mm_equation, mesh)
        filtered_strategies = []
        for strtg in mm_strategy.strategies:
            if strtg.input_specs is None:
                raise AssertionError(
                    f"Expected input_specs to be not None, got {strtg.input_specs}"
                )
            self_spec = strtg.input_specs[0]
            mat2_spec = strtg.input_specs[1]
            if is_tensor_shardable(
                self_strategy.shape, self_spec, allow_unbacked_sharding=True
            ) and is_tensor_shardable(
                mat2_strategy.shape, mat2_spec, allow_unbacked_sharding=True
            ):
                strtg.redistribute_cost = [
                    generate_redistribute_costs(self_strategy, self_spec),
                    generate_redistribute_costs(mat2_strategy, mat2_spec),
                ]
                filtered_strategies.append(strtg)
        mm_strategy.strategies = filtered_strategies
        return mm_strategy
else:
    from torch.distributed.tensor._ops._matrix_ops import _mm_like_strategy
