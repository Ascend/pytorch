from torch_npu._compat.version import CURRENT_VERSION

__all__ = [
    "register_op_strategy",
    "_mm_like_strategy",
    "_add_ephemeral_timeout_for_all_pgs",
]

# register_op_strategy moved from _ops.registration to _ops.utils in PyTorch 2.11.
from torch.distributed.tensor._ops.utils import register_op_strategy


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


# COMPAT(< 2026-08-04 torch nightly): the upstream
#   ``_add_ephemeral_timeout_for_all_pgs`` only became backend-generic
#   (dispatching through ``ProcessGroup::addEphemeralTimeout``, NPU included)
#   on the torch nightly of 2026-08-04 (pytorch#191980). On torch 2.13 and
#   earlier 2.14 nightlies it is CUDA/NCCL-only and a no-op on NPU, so the
#   torch_npu implementation must be used there.
# CAN REMOVE else branch when MIN_SUPPORTED >= the 2026-08-04 nightly.
if CURRENT_VERSION >= (2, 14):
    import inspect

    from torch.distributed import distributed_c10d as c10d

    _UPSTREAM_EPHEMERAL_TIMEOUT_IS_BACKEND_GENERIC = (
        "pg._add_ephemeral_timeout" in inspect.getsource(
            c10d._add_ephemeral_timeout_for_all_pgs)
    )
else:
    _UPSTREAM_EPHEMERAL_TIMEOUT_IS_BACKEND_GENERIC = False

if _UPSTREAM_EPHEMERAL_TIMEOUT_IS_BACKEND_GENERIC:
    from torch.distributed.distributed_c10d import _add_ephemeral_timeout_for_all_pgs
else:
    import torch
    from torch.distributed import distributed_c10d as c10d
    from datetime import timedelta

    def _add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
        """
        This API adds an ephemeral timeout extension for all PGs locally
        on one rank. The timeout gets reset when the first collective issued
        after API called finished.
        NOTE: We only support to set timeout for hccl backends for now.
        NOTE: While this feature provides flexibility in specific scenarios,
        it introduces statefulness
        to timeout setting. Therefore, it is advisable to use this API sparingly
        and consider alternative approaches, such as directly setting the timeout
        or utilizing a barrier collective (one can set any timeout to the barrier),
        whenever feasible.

        Args:
            timeout (timedelta): The delta of timeout to extend.

        Returns:
            None.
        """
        from torch_npu.distributed.distributed_c10d import is_hccl_available

        if not is_hccl_available():
            return

        try:
            from torch_npu._C._distributed_c10d import ProcessGroupHCCL
        except ImportError:
            return

        for pg in c10d._world.pg_map:
            devices = pg._device_types
            if torch.device("npu") in devices:
                backend = pg._get_backend(torch.device("npu"))
                if isinstance(backend, ProcessGroupHCCL) and hasattr(
                    backend, "_add_ephemeral_timeout"
                ):
                    backend._add_ephemeral_timeout(timeout)
    c10d._add_ephemeral_timeout_for_all_pgs = _add_ephemeral_timeout_for_all_pgs


# COMPAT(< 2.14): upstream ShardedTensor.cuda()/to() are CUDA-hardcoded on
#   torch < 2.14 (torch.cuda.current_device(), shard.tensor.cuda(), device
#   allowlist {"cuda", "xpu"}), so torch_npu patches a npu() method onto
#   ShardedTensor. Upstream pytorch#187939 (shipped in 2.14) makes cuda()/to()
#   hardware-agnostic via torch.accelerator, so the patch is not needed and
#   must not be applied on torch >= 2.14.
# CAN REMOVE this block when MIN_SUPPORTED >= (2, 14)
if CURRENT_VERSION < (2, 14):
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch_npu.distributed.tensor._sharded_tensor_patch import _patched_sharded_tensor_npu

    # Add the patched npu() method if it doesn't exist.
    if not hasattr(ShardedTensor, "npu"):
        ShardedTensor.npu = _patched_sharded_tensor_npu
