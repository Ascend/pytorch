from typing import Any, cast
import logging

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled, TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup, AllGatherState
from torch.distributed.utils import _to_kwargs

import torch_npu


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


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


def _get_param_all_gather_inputs(
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


def _patched_root_pre_forward(
    self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    self._lazy_init()
    if self._state_ctx.iter_forward_root is not None:
        return args, kwargs
    if not compiled_autograd_enabled():
        logger.debug("FSDP::root_pre_forward")
    self._state_ctx.iter_forward_root = self
    with torch.profiler.record_function("FSDP::root_pre_forward"):
        # Wait for optimizer before implicitly prefetched all-gathers
        event = self._state_ctx.post_optim_event
        if event is not None:
            self._comm_ctx.all_gather_copy_in_stream.wait_event(event)
            self._comm_ctx.all_gather_stream.wait_event(event)
            self._state_ctx.post_optim_event = None
        else:
            current_stream = self._device_handle.current_stream()
            self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
            self._comm_ctx.all_gather_stream.wait_stream(current_stream)
        # add patch for supporting self._device.type="npu"
        if self._device.type in ["cuda", "hpu", "xpu", "mtia", "npu"]:
            with torch.profiler.record_function("FSDP::inputs_to_device"):
                args_tuple, kwargs_tuple = _to_kwargs(
                    args, kwargs, self._device, False
                )  # same as DDP
            args, kwargs = args_tuple[0], kwargs_tuple[0]
    return args, kwargs


def _apply_fsdp_patch():
    FSDPParamGroup.finalize_backward = _patched_finalize_backward
    torch.distributed.fsdp._fully_shard._fsdp_collectives._get_param_all_gather_inputs = _get_param_all_gather_inputs
    torch.distributed.fsdp._fully_shard._fsdp_state.FSDPState._root_pre_forward = _patched_root_pre_forward
