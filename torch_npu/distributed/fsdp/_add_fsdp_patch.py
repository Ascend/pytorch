from typing import Tuple, Union, cast

import torch
from torch import distributed as dist
from torch._dynamo import tensor_version_op
from torch._prims import _make_prim, RETURN_TYPE
from torch.autograd.grad_mode import _unsafe_preserve_version_counter
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled, TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup, AllGatherState
from torch.profiler import record_function
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


def _patched_finalize_backward(self):
    self._wait_for_post_backward()
    for fsdp_param in self.fsdp_params:
        if fsdp_param.grad_offload_event is not None:
            fsdp_param.grad_offload_event.synchronize()
            fsdp_param.grad_offload_event = None
    if self._all_gather_result is not None:
        # If there was a mistargeted unshard without a corresponding wait,
        # then we wait here and clear the unshard
        if (event := self._all_gather_result.all_gather_event) is not None:
            torch.npu.current_stream().wait_event(event)
        work = self._all_gather_result.all_gather_work
        if isinstance(work, dist.distributed_c10d.Work):
            work.wait()
        self._all_gather_result = None
    self._post_forward_indices.clear()


class _patched_unsafe_preserve_version_counter(_unsafe_preserve_version_counter):
    r"""DO NOT USE THIS UNLESS YOU KNOW EXACTLY WHAT YOU'RE DOING.

    This context manager can lead to arbitrary silent-correctness issues in any other part of your code
    (even the ones not touched directly by the context manager)!

    Ordinarily, autograd will track mutations to tensors by incrementing it's `._version` attribute.
    This is generally important for correctness, as for example, mutating a tensor that autograd has saved
    for the backwards pass can result in incorrect gradients, and autograd uses the version counter to detect
    and error out in this situation.

    However, there are rare instances where it might be useful to hide mutations from autograd. For example:
    if a tensor is very large, and you'd like to free its memory by storing it elsewhere, and re-populate
    the tensor right before it is needed by autograd.

    Args:
        tensor (torch.Tensor): the tensor in question, that you would like to preserve the version counter of.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, tensors: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> None:
        self.tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tensors
        if not isinstance(self.tensors, tuple):
            raise TypeError("Input must be tuple tensors." + pta_error(ErrCode.TYPE))
        self.prev_versions = tuple(t._version for t in self.tensors)

    def __exit__(self, *args) -> None:
        torch_npu._C._unsafe_set_npu_version_counter(self.tensors, self.prev_versions)


_patched_unsafe_set_npu_version_counter = _make_prim(
    schema="_unsafe_set_npu_version_counter(Tensor[] tensors, SymInt[] versions) -> ()",
    return_type=RETURN_TYPE.NEW,
    meta=lambda self, version: None,
    impl_aten=torch_npu._C._unsafe_set_npu_version_counter,
    doc="Tracable+SymInt version of torch_npu._C._unsafe_set_npu_version_counter",
)


def _patched_unsafe_set_version_counter_functional(ctx, tensors, version):
    torch_npu._C._unsafe_set_npu_version_counter(tensors, version)


@torch.no_grad()
def foreach_all_gather_copy_out_npu(
        all_gather_result: AllGatherResult,
        fsdp_params: list[FSDPParam],
        group: dist.ProcessGroup,
) -> None:
    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result
    _dtype, device = all_gather_output.dtype, all_gather_output.device
    device_handle = _get_device_handle(device.type)
    if all_gather_event is not None:  # sync op
        device_handle.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()
    world_size, device = group.size(), all_gather_output.device
    split_with_sizes_out: list[torch.Tensor] = []
    shard_i_copy_infos: list[tuple[FSDPParam, list[torch.Tensor]]] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
            param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        # NOTE: Under compile, make sure we always recreate all_gather_outputs
        # per AllGather. See [Note: Invariants for torch.compile Traceable FSDP2].
        force_recreate = compiled_autograd_enabled()
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels,
            all_gather_input_dtypes,
            world_size,
            device,
            force_recreate=force_recreate,
        )
        if not force_recreate:
            fsdp_param.alloc_all_gather_outputs()
        param_all_gather_outputs = fsdp_param.all_gather_outputs
        if fsdp_param.fsdp_placement.dim != 0:
            # Copy to a temporary and then chunk-cat into the final all-gather
            # output tensors
            param_all_gather_outputs = [
                torch.empty_like(t) for t in param_all_gather_outputs
            ]
            shard_i_copy_infos.append((fsdp_param, param_all_gather_outputs))
        split_with_sizes_out.extend(param_all_gather_outputs)
    all_gather_output = all_gather_output.view(world_size, -1)
    if all_gather_output.dtype == torch.uint8:
        out = [t.view(world_size, -1).view(torch.uint8) for t in split_with_sizes_out]
    else:
        out = [t.view(world_size, -1) for t in split_with_sizes_out]
    with torch.autograd._unsafe_preserve_version_counter(tuple(out)):
        torch.ops.fsdp.split_with_sizes_copy(
            all_gather_output, all_gather_input_split_sizes, dim=1, out=out
        )

    for fsdp_param, param_all_gather_outputs in shard_i_copy_infos:
        # Chunk-cat from the temporary to the final all-gather output tensors
        shard_dim = fsdp_param.fsdp_placement.dim

        with torch.autograd._unsafe_preserve_version_counter(
                tuple(fsdp_param.all_gather_outputs)
        ):
            for param_all_gather_output, target_all_gather_output in zip(
                    param_all_gather_outputs, fsdp_param.all_gather_outputs
            ):
                padded_sharded_size = (
                    fsdp_param.padded_sharded_param_size
                    if fsdp_param.sharded_state == ShardedState.SHARDED
                    else cast(
                        torch.Tensor, fsdp_param._sharded_post_forward_param_data
                    ).size()
                )
                pre_param_size = list(padded_sharded_size)
                pre_param_size[0] *= world_size
                chunks = torch.chunk(
                    param_all_gather_output.view(pre_param_size), world_size, dim=0
                )
                post_param_size = list(padded_sharded_size)
                post_param_size[shard_dim] *= world_size
                cat_out = target_all_gather_output.view(post_param_size)
                torch.cat(chunks, dim=shard_dim, out=cat_out)


def patched_wait_for_unshard(self):
    """
    1. In forward with implict prefetching, to overlap the current copy-out
    with the next all-gather, we save a reference to the current all-gather
    result to free after the next copy-out.
    2. Otherwise (explicit prefetching or in backward), we free the
    all-gather result immediately after the current copy-out since we can
    already overlap the current copy-out with the previous reduce-scatter.
    """
    if not self._all_gather_result:
        return  # no preceding unshard
    async_op = self._all_gather_result.all_gather_work is not None
    if self._training_state == TrainingState.FORWARD:  # implicit prefetch
        if prev_all_gather_state := self.comm_ctx.all_gather_state:
            self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
            self.comm_ctx.all_gather_state = None  # free the all-gather result
    with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
        foreach_all_gather_copy_out_npu(
            self._all_gather_result,
            self.fsdp_params,
            self._all_gather_process_group,
        )
    for fsdp_param in self.fsdp_params:
        fsdp_param.init_unsharded_param()
    self._to_unsharded()
    all_gather_copy_out_event = self.device_handle.Event()
    all_gather_copy_out_event.record()
    if not async_op and self._training_state == TrainingState.FORWARD:
        # Defer free to allow for overlap of this copy-out with next
        # all-gather collective
        self.comm_ctx.all_gather_state = AllGatherState(
            self._all_gather_result, all_gather_copy_out_event
        )
    else:
        self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
    self._all_gather_result = None  # free unless saved in `all_gather_state`


def _apply_fsdp_patch():
    FSDPParamGroup.finalize_backward = _patched_finalize_backward
    FSDPParamGroup.wait_for_unshard = patched_wait_for_unshard
    tensor_version_op._unsafe_set_version_counter_functional = _patched_unsafe_set_version_counter_functional
    tensor_version_op._unsafe_set_version_counter = _patched_unsafe_set_npu_version_counter
    _unsafe_preserve_version_counter.__init__ = _patched_unsafe_preserve_version_counter.__init__
    _unsafe_preserve_version_counter.__exit__ = _patched_unsafe_preserve_version_counter.__exit__
