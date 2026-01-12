import copy

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard


def _patched_sharded_tensor_npu(
        self,
        device=None,
        non_blocking=False,
        memory_format=torch.preserve_format,
        process_group=None,
) -> ShardedTensor:
    """
    Returns a copy of this object in NPU memory, if the original ShardedTensor
    is on CPU, we will move the local shard to the current NPU device of each
    process in a SPMD fashion.
    If this ShardedTensor is already on NPU memory and local shards on each rank are
    already on current device, we still returns a new ShardedTensor object with new
    metadata, but no underlying data movements are performed.

    .. note:: When moving a ShardedTensor from CPU to NPU, the ShardedTensor might
        need to be managed by a different type of ProcessGroup that is compatible
        with NPU, it is the user's responsibility to explicitly pass in a new
        process_group that is compatible with NPU.

    Args:
        device (torch.device/str, optional): Target NPU device (only "npu" without index is supported).
            Defaults to None, which uses current NPU device.
        non_blocking (bool, optional): If True, the copy is asynchronous with respect to the host.
            Defaults to False.
        memory_format (torch.memory_format, optional): Desired memory format of the tensor after move.
            Only torch.preserve_format or torch.contiguous_format is supported. Defaults to torch.preserve_format.
        process_group (ProcessGroup, optional): The process group to use for the new ShardedTensor.
            Defaults to None, which uses the original process group of the ShardedTensor.

    Returns:
        ShardedTensor: A new ShardedTensor instance with all local shards moved to NPU device

    Raises:
        RuntimeError: If memory_format is not torch.preserve_format or torch.contiguous_format
        ValueError: If device is specified but not a valid NPU device without index
    """
    if (
            memory_format != torch.preserve_format
            and memory_format != torch.contiguous_format
    ):
        raise RuntimeError(
            "Only `torch.contiguous_format` or "
            "`torch.preserve_format` is supported!"
        )

    current_device = torch.device(torch.npu.current_device())
    # returns a copy of ShardedTensor on NPU current device
    list_shards: list[Shard] = []
    # move all local shards to current device, and change metadata
    # if local shards already on the current device, there's no
    # real data movement, only the metadata are copied.
    for shard in self._local_shards:
        npu_tensor = shard.tensor.npu(
            device=current_device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )  # type: ignore[call-arg]
        metadata = copy.deepcopy(shard.metadata)
        metadata.placement._device = current_device  # type: ignore[union-attr]

        list_shards.append(Shard(npu_tensor, metadata))

    st_meta = copy.deepcopy(self.metadata())
    for meta in st_meta.shards_metadata:
        if meta.placement.device().type != "npu":  # type: ignore[union-attr]
            meta.placement._device = current_device  # type: ignore[union-attr]

    pg = self._process_group if process_group is None else process_group
    # we need to use `init_from_local_shards` to communicate between ranks
    # and update the sharding spec/shards metadata.
    st_npu = ShardedTensor._init_from_local_shards_and_global_metadata(
        list_shards,
        sharded_tensor_metadata=st_meta,
        process_group=pg,
        init_rrefs=self._init_rrefs,
    )
    return st_npu


def _apply_sharded_tensor_npu_patch():
    """
    Applies the NPU patch to ShardedTensor by adding/overriding the npu() method.
    """
    if not hasattr(ShardedTensor, "npu"):
        ShardedTensor.npu = _patched_sharded_tensor_npu


# Execute the patch application when the module is imported
_apply_sharded_tensor_npu_patch()