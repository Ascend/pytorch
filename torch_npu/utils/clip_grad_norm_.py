import functools
import math
import warnings
from typing import (
    List,
    Union,
)
import torch
import torch.distributed as dist
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.fully_sharded_data_parallel import _get_grad_norm


__all__ = []


@torch.no_grad()
def _clip_grad_norm_(
    self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
) -> torch.Tensor:
    _lazy_init(self, self)
    if not self._is_root:
        raise RuntimeError(
            "`clip_grad_norm_()` should only be called on the root FSDP instance"
        )
    self._assert_state(TrainingState.IDLE)
    # If every FSDP instance uses `NO_SHARD`, then we can directly use
    # the normal `nn.utils` one targeting local gradients
    all_no_shard = all(
        not handle.uses_sharded_strategy for handle in self._all_handles
    )
    if all_no_shard:
        return torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm, norm_type
        )
    # Otherwise, there exists some FSDP instance using a sharded strategy,
    # where sharded and non-sharded parameters must be handled separately
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    sharded_params_set = set()
    nonsharded_params_set = set()  # `NO_SHARD` or not FSDP-managed
    # Make sure to compute the local norm using lists for deterministic
    # iteration order and hence deterministic total norm computation
    sharded_params = []
    nonsharded_params = []
    grads: List[torch.Tensor] = []
    for handle in self._all_handles:
        if handle.uses_sharded_strategy:
            target_set = sharded_params_set
            target_list = sharded_params
        else:
            target_set = nonsharded_params_set
            target_list = nonsharded_params
        if handle._use_orig_params:
            for param in handle.flat_param._params:
                if param not in target_set:
                    target_set.add(param)
                    target_list.append(param)
                    if param.grad is not None:
                        grads.append(param.grad)
        else:
            if handle.flat_param not in target_set:
                target_set.add(handle.flat_param)
                target_list.append(handle.flat_param)
                if handle.flat_param.grad is not None:
                    grads.append(handle.flat_param.grad)
    for param in self.parameters():
        not_fsdp_managed = (
            param not in sharded_params_set and param not in nonsharded_params_set
        )
        if not_fsdp_managed:
            nonsharded_params_set.add(param)
            nonsharded_params.append(param)
            if param.grad is not None:
                grads.append(param.grad)
    # Compute local norms (forced to be in FP32)
    local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(
        self.compute_device
    )
    local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(
        self.compute_device
    )
    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
        dist.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group
        )
    else:
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=self.process_group)
        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        total_norm += local_nonsharded_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    if self.cpu_offload.offload_params:
        total_norm = total_norm.cpu()

    clip_coef = max_norm / (total_norm + 1e-6)
    # Multiplying by the clamped coefficient is meaningless when it is
    # equal to 1, but it avoids the host-device sync that would result from
    # `if clip_coef < 1`
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.mul_(clip_coef_clamped.to(grad.device, grad.dtype))
    # Use the "largest" dtype by type promotion semantics to use the same
    # dtype as if we did not force local norm computation to be in FP32
    if len(grads) == 0:
        # If this rank has no gradients, then we must default to FP32
        # unless we use additional communication, which we prefer to avoid
        # since `clip_grad_norm_()` is called in the training loop
        warnings.warn(
            f"Called FSDP.clip_grad_norm_() on rank {self.rank} with no "
            "gradients -- returning the total norm in the default dtype "
            f"{total_norm.dtype}"
        )  # warn since this is generally unexpected
        return total_norm
    total_norm_dtype = functools.reduce(
        torch.promote_types,
        [grad.dtype for grad in grads],
    )
    return total_norm.to(total_norm_dtype)


def _apply_clip_grad_norm_patch():
    torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.clip_grad_norm_ = _clip_grad_norm_
