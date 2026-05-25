import operator
from collections import defaultdict
from collections.abc import Sequence
from functools import reduce
from typing import Optional, Union

import torch
from torch.distributed.fsdp import fully_shard as torch_fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState


_FSDP_ENHANCE_PATCH_APPLIED = False


class FSDPMemCache:
    def __init__(self):
        self.buffers = defaultdict(list)  # dtype -> buffer list
        self.used = defaultdict(list)  # dtype -> bool list

    def _get_storage_ptr(self, tensor: torch.Tensor) -> int:
        return tensor.storage().data_ptr()

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
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


def _patched_fsdp_param_group_init(original_func):
    def wrapper(self, *args, **kwargs):
        original_func(self, *args, **kwargs)
        # set _use_mem_cache to fsdp_params
        if self.modules and self.fsdp_params:
            use_mem_cache = getattr(self.modules[0], "_use_mem_cache", False)
            self._all_gather_comm._use_mem_cache = use_mem_cache
            self._reduce_scatter_comm._use_mem_cache = use_mem_cache

    return wrapper


def _patched_all_gather_allocate(
    self,
    size: Sequence[Union[int, torch.SymInt]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if getattr(self, "_use_mem_cache", False):
        return _fsdp_mem_cache.allocate(size, dtype=dtype, device=device)
    return torch.empty(*size, dtype=dtype, device=device)


def _patched_wait_all_gather_streams_on_event(original_func):
    def wrapper(self, event: Optional[torch.Event]):
        original_func(self, event)
        # if previous layer deferred free for overlap, free its output in current comm_ctx.all_gather_state
        if (
            self._training_state == TrainingState.FORWARD
            and self.comm_ctx.all_gather_state
        ):
            prev_all_gather_output = (
                self.comm_ctx.all_gather_state.all_gather_result.all_gather_output
            )
            _fsdp_mem_cache.free(prev_all_gather_output)
        # if current layer no need to defer free, free output after all_gather_copy_out event
        elif self._all_gather_result:
            all_gather_output = self._all_gather_result.all_gather_output
            _fsdp_mem_cache.free(all_gather_output)

    return wrapper


def _patched_reduce_scatter_allocate(
    self,
    size: Sequence[Union[int, torch.SymInt]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # foreach_reduce allocates memory for both input and output, we only cache the input, i.e. on the first call
    if getattr(self, "_use_mem_cache", False) and not getattr(
        self, "_mem_cache_flag", False
    ):
        self._mem_cache_flag = True
        return _fsdp_mem_cache.allocate(size, dtype=dtype, device=device)
    return torch.empty(*size, dtype=dtype, device=device)


def _patched_foreach_reduce(original_foreach_reduce):
    def wrapper(*args, **kwargs):
        out = original_foreach_reduce(*args, **kwargs)
        # free memory cache for reduce-scatter input
        _fsdp_mem_cache.free(out[0])
        reduce_scatter_comm = kwargs.get("reduce_scatter_comm", args[4])
        reduce_scatter_comm._mem_cache_flag = False
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
        setattr(src_obj, src_attr, type(src_value)())  # [] / () / set() / {}
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
                move_attr(
                    self, "_backup_forward_fetch", self, "_states_to_forward_prefetch"
                )
            return original_post_forward(self, module, args, out)

        # backup forward prefetch state before original post_forward
        move_attr(self, "_states_to_forward_prefetch", self, "_backup_forward_fetch")
        return original_post_forward(self, module, args, out)

    return wrapper


def _apply_fsdp_enhance_patch():
    global _FSDP_ENHANCE_PATCH_APPLIED
    if _FSDP_ENHANCE_PATCH_APPLIED:
        return

    # support using memory cache for FSDP comm ops
    FSDPParamGroup.__init__ = _patched_fsdp_param_group_init(FSDPParamGroup.__init__)
    FSDPParamGroup._wait_all_gather_streams_on_event = (
        _patched_wait_all_gather_streams_on_event(
            FSDPParamGroup._wait_all_gather_streams_on_event
        )
    )
    torch.distributed.fsdp._fully_shard._fsdp_collectives.DefaultAllGather.allocate = (
        _patched_all_gather_allocate
    )
    torch.distributed.fsdp._fully_shard._fsdp_collectives.DefaultReduceScatter.allocate = _patched_reduce_scatter_allocate
    origin_foreach_reduce = (
        torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce
    )
    torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce = (
        _patched_foreach_reduce(origin_foreach_reduce)
    )
    # _fsdp_param_group imported these functions before patching
    torch.distributed.fsdp._fully_shard._fsdp_param_group.DefaultAllGather.allocate = (
        _patched_all_gather_allocate
    )
    torch.distributed.fsdp._fully_shard._fsdp_param_group.DefaultReduceScatter.allocate = _patched_reduce_scatter_allocate
    torch.distributed.fsdp._fully_shard._fsdp_param_group.foreach_reduce = (
        _patched_foreach_reduce(origin_foreach_reduce)
    )

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
