from typing import cast

import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.distributed.fsdp._fully_shard._fsdp_collectives import allocate_memory
from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled, TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup, AllGatherState

import torch_npu


def _patched_finalize_backward(self):
    self._wait_for_post_backward()
    for fsdp_param in self.fsdp_params:
        if fsdp_param.grad_offload_event is not None:
            fsdp_param.grad_offload_event.synchronize()
            fsdp_param.grad_offload_event = None
    if self._all_gather_result is not None:
        # If there was a mistargeted unshard without a corresponding wait,
        # then we wait here and clear the unshard
        event = self._all_gather_result.all_gather_event
        if event is not None:
            torch.npu.current_stream().wait_event(event)
        work = self._all_gather_result.all_gather_work
        if isinstance(work, dist.distributed_c10d.Work):
            work.wait()
        self._all_gather_result = None
    self._post_forward_indices.clear()


def _patched_get_param_all_gather_inputs(
    fsdp_params: list[FSDPParam],
) -> list[list[torch.Tensor]]:
    if compiled_autograd_enabled():
        return [fsdp_param.all_gather_inputs for fsdp_param in fsdp_params]

    # Intentionally try to run a fast-path that bypasses abstractions for the
    # common FSDP case of bf16/fp32 mixed precision in order to use foreach
    # copy for lower CPU overhead and more efficient copying in eager
    def use_foreach_copy(fsdp_param: FSDPParam) -> bool:
        return (
            fsdp_param.param_dtype is not None
            and not fsdp_param.offload_to_cpu
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        )

    param_all_gather_inputs: list[list[torch.Tensor]] = [[] for _ in fsdp_params]
    foreach_copy_indices: list[int] = []
    foreach_copy_inputs: list[torch.Tensor] = []
    foreach_copy_input_numels: list[int] = []

    # 1st pass: for foreach-copy parameters, get inputs and metadata for the
    # foreach copy, and for the others, actually get their all-gather inputs
    for i, fsdp_param in enumerate(fsdp_params):
        if use_foreach_copy(fsdp_param):
            foreach_copy_indices.append(i)
            all_gather_input = (
                fsdp_param._sharded_param_data
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data)
            )
            foreach_copy_inputs.append(all_gather_input)
            foreach_copy_input_numels.append(all_gather_input.numel())
        else:
            param_all_gather_inputs[i] = fsdp_param.all_gather_inputs

    # 2nd pass: use foreach copy to compute the remaining all-gather inputs
    if foreach_copy_inputs:
        fsdp_param_0 = fsdp_params[foreach_copy_indices[0]]
        param_dtype, device = fsdp_param_0.param_dtype, fsdp_param_0.device
        flat_foreach_copy_input = torch.empty(
            (sum(foreach_copy_input_numels),), device=device, dtype=param_dtype
        )
        splits = torch.split(flat_foreach_copy_input, foreach_copy_input_numels)
        # patch in npu: set non_blocking=True
        if splits[0].device == foreach_copy_inputs[0].device:
            torch._foreach_copy_(splits, foreach_copy_inputs, non_blocking=True)
        else:
            torch._foreach_copy_(splits, foreach_copy_inputs)
        for i, split in zip(foreach_copy_indices, splits):
            param_all_gather_inputs[i] = [split]

    return param_all_gather_inputs


def _patched_all_gather_copy_in(
    all_gather_inputs: list[torch.Tensor],
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
    allocate_memory_from_process_group: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_gather_output = allocate_memory(
        all_gather_input_numel * world_size,
        dtype=dtype,
        device=device,
        group=_resolve_process_group(group_name),
        from_process_group=allocate_memory_from_process_group,
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        # patch in npu: set non_blocking=True
        if foreach_copy_dsts[0].device == all_gather_inputs[0].device:
            torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs, non_blocking=True)
        else:
            torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


def _apply_fsdp_patch():
    FSDPParamGroup.finalize_backward = _patched_finalize_backward
    torch.distributed.fsdp._fully_shard._fsdp_collectives._get_param_all_gather_inputs \
        = _patched_get_param_all_gather_inputs
    torch.ops.fsdp.all_gather_copy_in = _patched_all_gather_copy_in
