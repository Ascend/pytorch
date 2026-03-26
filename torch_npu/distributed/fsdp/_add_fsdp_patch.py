from collections import defaultdict
from functools import reduce
from typing import cast, Optional, Sequence, Union
from weakref import WeakKeyDictionary
import operator

import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.distributed.fsdp import fully_shard as torch_fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled, TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

import torch_npu


_FSDP_ENHANCE_PATCH_APPLIED = False


class FSDPMemCache:
    def __init__(self):
        self.buffers = defaultdict(list) # dtype -> buffer list
        self.used = defaultdict(list)    # dtype -> bool list
        # use WeakKeyDictionary to identify when using cache, since ProcessGroup does not support custom attributes
        self.pg_attrs: WeakKeyDictionary[dist.ProcessGroup, dict] = WeakKeyDictionary()

    def set_pg_attr(self, pg: dist.ProcessGroup, key: str, value: Union[bool, None]):
        self.pg_attrs.setdefault(pg, {})[key] = value

    def get_pg_attr(self, pg: dist.ProcessGroup, key: str, default=None):
        return self.pg_attrs.get(pg, {}).get(key, default)

    def _get_storage_ptr(self, tensor: torch.Tensor) -> int:
        return tensor.storage().data_ptr()

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
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


def _patched_allocate_memory(
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    from_process_group: bool,
) -> torch.Tensor:
    if from_process_group:
        backend = group._get_backend(device)
        if backend.supports_tensor_alloc(device):
            return backend.allocate_tensor(size, dtype=dtype, device=device)
    if not _FSDP_ENHANCE_PATCH_APPLIED or not _fsdp_mem_cache.get_pg_attr(group, "_mem_cache_flag", None):
        return torch.empty((size,), dtype=dtype, device=device)
    # foreach_all_gather calls this function once to allocate output
    # foreach_reduce calls this function twice to allocate input and output, we only cache the input
    _fsdp_mem_cache.set_pg_attr(group, "_mem_cache_flag", False)
    return _fsdp_mem_cache.allocate((size,), dtype=dtype, device=device)


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
    all_gather_output = _patched_allocate_memory(
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


def _patched_fsdp_param_group_init(original_func):
    def wrapper(self, *args, **kwargs):
        original_func(self, *args, **kwargs)
        # set _use_mem_cache to fsdp_params
        if self.modules and self.fsdp_params:
            use_mem_cache = getattr(self.modules[0], "_use_mem_cache", False)
            for fsdp_param in self.fsdp_params:
                fsdp_param._use_mem_cache = use_mem_cache
    return wrapper


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


def _patched_foreach_all_gather(original_func):
    def wrapper(*args, **kwargs):
        fsdp_params = kwargs.get("fsdp_params", args[0])
        if not fsdp_params or not getattr(fsdp_params[0], "_use_mem_cache", False):
            return original_func(*args, **kwargs)

        group = kwargs.get("group", args[1])
        _fsdp_mem_cache.set_pg_attr(group, "_mem_cache_flag", True)
        out = original_func(*args, **kwargs)
        _fsdp_mem_cache.set_pg_attr(group, "_mem_cache_flag", None)
        return out
    return wrapper


def _patched_foreach_reduce(original_foreach_reduce):
    def wrapper(*args, **kwargs):
        fsdp_params = kwargs.get("fsdp_params", args[0])
        if not fsdp_params or not getattr(fsdp_params[0], "_use_mem_cache", False):
            return original_foreach_reduce(*args, **kwargs)

        reduce_scatter_group = kwargs.get("reduce_scatter_group", args[2])
        _fsdp_mem_cache.set_pg_attr(reduce_scatter_group, "_mem_cache_flag", True)
        out = original_foreach_reduce(*args, **kwargs)
        # free memory cache for reduce-scatter input
        _fsdp_mem_cache.free(out[0])
        _fsdp_mem_cache.set_pg_attr(reduce_scatter_group, "_mem_cache_flag", None)
        return out
    return wrapper


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
    # essential patch to run on NPU
    FSDPParamGroup.finalize_backward = _patched_finalize_backward
    torch.distributed.fsdp._fully_shard._fsdp_collectives._get_param_all_gather_inputs \
        = _patched_get_param_all_gather_inputs
    torch.ops.fsdp.all_gather_copy_in = _patched_all_gather_copy_in
    torch.ops.fsdp.all_gather_copy_in.default = _patched_all_gather_copy_in


def _apply_fsdp_enhance_patch():
    global _FSDP_ENHANCE_PATCH_APPLIED
    if _FSDP_ENHANCE_PATCH_APPLIED:
        return

    # support using memory cache for FSDP comm ops
    torch.distributed.fsdp._fully_shard._fsdp_collectives.allocate_memory = _patched_allocate_memory
    FSDPParamGroup.__init__ = _patched_fsdp_param_group_init(FSDPParamGroup.__init__)
    FSDPParamGroup._wait_all_gather_streams_on_event \
        = _patched_wait_all_gather_streams_on_event(FSDPParamGroup._wait_all_gather_streams_on_event)
    origin_foreach_all_gather = torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather
    torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather \
        = _patched_foreach_all_gather(origin_foreach_all_gather)
    origin_foreach_reduce = torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce
    torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce \
        = _patched_foreach_reduce(origin_foreach_reduce)
    # _fsdp_param_group imported these functions before patching
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_all_gather \
        = _patched_foreach_all_gather(origin_foreach_all_gather)
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce \
        = _patched_foreach_reduce(origin_foreach_reduce)

    # optimize communication, e.g. removing redundant all-gather when recomputing in backward
    FSDPState._post_forward = _patched_fsdp_state_post_forward(FSDPState._post_forward)
    FSDPParamGroup.post_forward = _patched_post_forward(FSDPParamGroup.post_forward)
    FSDPParamGroup.post_backward = _patched_post_backward(FSDPParamGroup.post_backward)

    _FSDP_ENHANCE_PATCH_APPLIED = True


def fully_shard(*args, **kwargs):
    _apply_fsdp_enhance_patch()
    return torch_fully_shard(*args, **kwargs)


fully_shard.state = torch_fully_shard.state
fully_shard.__doc__ = torch_fully_shard.__doc__
