from typing import cast, Any, Callable, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
from functools import reduce
from itertools import chain
import logging
import operator

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _div_if_needed,
    _get_all_gather_input_metadatas,
    _get_gradient_divide_factors,
    foreach_reduce_scatter_copy_in,
    AllGatherResult
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
    TrainingState,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.distributed.utils import _to_kwargs

import torch_npu


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


class FSDPMemCache:
    def __init__(self):
        self.buffers = defaultdict(list) # dtype -> buffer list
        self.used = defaultdict(list)    # dtype -> bool list

    def _get_storage_ptr(self, tensor: torch.Tensor) -> int:
        return tensor.storage().data_ptr()

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        buffer_list = self.buffers[dtype]
        used = self.used[dtype]
        for i, buffer in enumerate(buffer_list):
            if not used[i]:
                needed_numel = reduce(operator.mul, size)
                if buffer.numel() < needed_numel:
                    buffer = torch.empty(size, dtype=dtype, device=device)
                    buffer_list[i] = buffer
                used[i] = True
                return buffer[:needed_numel].view(size)

        buffer = torch.empty(size, dtype=dtype, device=device)
        buffer_list.append(buffer)
        used.append(True)
        return buffer

    def free(self, tensor: torch.Tensor):
        # if tensor is not from this cache, do nothing
        buffer_list = self.buffers[tensor.dtype]
        storage_ptr = self._get_storage_ptr(tensor)
        for i, buffer in enumerate(buffer_list):
            if self._get_storage_ptr(buffer) == storage_ptr:
                self.used[tensor.dtype][i] = False
                return

    def clear(self):
        self.buffers.clear()
        self.used.clear()


_fsdp_mem_cache = FSDPMemCache()


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


def _patched_fsdp_param_group_init(original_func):
    def wrapper(self, *args, **kwargs):
        original_func(self, *args, **kwargs)
        # set _use_mem_cache to fsdp_params
        if self.modules and self.fsdp_params:
            use_mem_cache = getattr(self.modules[0], "_use_mem_cache", False)
            for fsdp_param in self.fsdp_params:
                fsdp_param._use_mem_cache = use_mem_cache
    return wrapper


def _custom_all_gather_copy_in(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_gather_output = _fsdp_mem_cache.allocate(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        if foreach_copy_dsts[0].device == all_gather_inputs[0].device:
            torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs, non_blocking=True)
        else:
            torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


@torch.no_grad()
def _patched_foreach_all_gather(
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.Stream,
    all_gather_stream: torch.Stream,
    device: torch.device,
) -> Optional[AllGatherResult]:
    world_size, rank = group.size(), group.rank()
    device_handle = _get_device_handle(device.type)
    with device_handle.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype == torch.uint8:
            all_gather_inputs = [
                t.view(torch.uint8)
                for ts in param_all_gather_inputs
                for t in ts
            ]
        else:
            all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        # patch in npu: use memory cache for all-gather output
        use_mem_cache = fsdp_params[0]._use_mem_cache if fsdp_params else False
        if use_mem_cache:
            all_gather_input, all_gather_output = _custom_all_gather_copy_in(
                all_gather_inputs, inp_split_sizes, all_gather_input_numel, world_size, rank, dtype, device
            )
        else:
            all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
                all_gather_inputs,
                inp_split_sizes,
                all_gather_input_numel,
                world_size,
                rank,
                dtype,
                device,
            )
        del param_all_gather_inputs
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = all_gather_stream.record_event()
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


def _patched_wait_all_gather_streams_on_event(original_func):
    def wrapper(self, event: Optional[torch.Event]):
        original_func(self, event)
        # if previous layer deferred free for overlap, free its output in current comm_ctx.all_gather_state
        if self._training_state == TrainingState.FORWARD and self.comm_ctx.all_gather_state:
            prev_all_gather_output = self.comm_ctx.all_gather_state.all_gather_result.all_gather_output
            _fsdp_mem_cache.free(prev_all_gather_output)
        # if current layer no need to defer free, free output after all_gather_copy_out event
        elif self._all_gather_result:
            all_gather_output = self._all_gather_result.all_gather_output
            _fsdp_mem_cache.free(all_gather_output)
    return wrapper


@torch.no_grad()
def _patched_foreach_reduce(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    reduce_scatter_reduce_op: Optional[Union[dist.ReduceOp, dist.ReduceOp.RedOpType]],
    all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
    all_reduce_hook: Optional[Callable[[torch.Tensor], None]],
) -> tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    Optional[torch.Tensor],
    Optional[torch.Event],
    Optional[torch.Tensor],
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    predivide_factor, postdivide_factor = _get_gradient_divide_factors(
        reduce_scatter_group, all_reduce_group, reduce_dtype, device.type
    )
    world_size = reduce_scatter_group.size()
    for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
        if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
            continue
        chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
        unsharded_grads[i] = torch.cat(chunks, dim=0)
    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    # patch in npu: use memory cache for reduce-scatter input
    use_mem_cache = fsdp_params[0]._use_mem_cache if fsdp_params else False
    if use_mem_cache:
        reduce_scatter_input = _fsdp_mem_cache.allocate(
            (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
        )
    else:
        reduce_scatter_input = torch.empty(
            (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
        )
    device_handle = _get_device_handle(device.type)
    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
    current_stream = device_handle.current_stream()
    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None
    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if reduce_scatter_reduce_op is None:
            if predivide_factor is None:
                reduce_scatter_reduce_op = ReduceOp.AVG
            else:
                reduce_scatter_reduce_op = ReduceOp.SUM
        dist.reduce_scatter_tensor(
            output=reduce_output,
            input=reduce_scatter_input,
            group=reduce_scatter_group,
            op=reduce_scatter_reduce_op,
        )
        # patch in npu: free reduce-scatter input for reuse
        if use_mem_cache:
            _fsdp_mem_cache.free(reduce_scatter_input)
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
                )
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
            if to_accumulate_grad:
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (
                    getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
                    or {}
                ).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


def _patched_post_forward(original_post_forward):
    # if _skip_post_forward is True, return output directly and skip the original post_forward
    # if False, run original post_forward and then set _skip_post_forward to True
    def wrapper(self, module, args, out):
        if getattr(self, "_skip_post_forward", False):
            return out
        out = original_post_forward(self, module, args, out)
        self._skip_post_forward = True
        return out
    return wrapper


def _patched_post_backward(original_post_backward):
    # reset _skip_post_forward to False before running original post_backward
    def wrapper(self):
        self._skip_post_forward = False
        original_post_backward(self)
    return wrapper


def move_attr(src_obj, src_attr, dst_obj, dst_attr):
    # move value of src_attr from src_obj to dst_attr of dst_obj, and clear src_attr of src_obj
    src_value = getattr(src_obj, src_attr, None)
    setattr(dst_obj, dst_attr, src_value)
    if type(src_value) in (list, tuple, set, dict):
        setattr(src_obj, src_attr, type(src_value)()) # [] / () / set() / {}
    else:
        setattr(src_obj, src_attr, None)


def _patched_fsdp_state_post_forward(original_post_forward):
    # manage forward prefetch state backup/restoration based on training state
    # skip custom logic if skip flag is set or training state is PRE_BACKWARD
    def wrapper(self, module, args, out):
        if not hasattr(self, "_backup_forward_fetch"):
            self._backup_forward_fetch = None

        if hasattr(module, "skip_custom_post_forward") or not self._fsdp_param_group:
            return original_post_forward(self, module, args, out)

        # restore backup state if _skip_post_forward is True or in PRE_BACKWARD state
        skip_post_forward = getattr(self._fsdp_param_group, "_skip_post_forward", False)
        if skip_post_forward or self._training_state == TrainingState.PRE_BACKWARD:
            if self._backup_forward_fetch is not None:
                move_attr(self, "_backup_forward_fetch", self, "_states_to_forward_prefetch")
            return original_post_forward(self, module, args, out)

        # backup forward prefetch state before original post_forward
        move_attr(self, "_states_to_forward_prefetch", self, "_backup_forward_fetch")
        return original_post_forward(self, module, args, out)
    return wrapper


def _apply_fsdp_patch():
    FSDPState._post_forward = _patched_fsdp_state_post_forward(FSDPState._post_forward)
    FSDPParamGroup.__init__ = _patched_fsdp_param_group_init(FSDPParamGroup.__init__)
    FSDPParamGroup._wait_all_gather_streams_on_event \
        = _patched_wait_all_gather_streams_on_event(FSDPParamGroup._wait_all_gather_streams_on_event)
    FSDPParamGroup.post_forward = _patched_post_forward(FSDPParamGroup.post_forward)
    FSDPParamGroup.post_backward = _patched_post_backward(FSDPParamGroup.post_backward)
    FSDPParamGroup.finalize_backward = _patched_finalize_backward
    torch.distributed.fsdp._fully_shard._fsdp_collectives._get_param_all_gather_inputs = _get_param_all_gather_inputs
    torch.distributed.fsdp._fully_shard._fsdp_state.FSDPState._root_pre_forward = _patched_root_pre_forward
    torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather = _patched_foreach_all_gather
    torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce = _patched_foreach_reduce
    # _fsdp_param_group imported these functions before patching
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather = _patched_foreach_all_gather
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce = _patched_foreach_reduce
