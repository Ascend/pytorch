import torch
from torch.distributed import _rank_not_in_group, BarrierOptions
from torch.distributed.distributed_c10d import (
    get_backend, _get_default_group, exception_handler, GroupMember,
    _warn_not_in_group
)
from torch._C._distributed_c10d import BarrierOptions

__all__ = ['barrier']


@exception_handler
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None):
    """
    Synchronizes all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        device_ids ([int], optional): List of device ids.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("barrier")
        return

    opts = BarrierOptions()
    if device_ids is not None:
        if isinstance(device_ids, list):
            opts.device_ids = device_ids
        else:
            raise RuntimeError(
                "Invalid function argument: " "device_ids type should be List[int]")

    if group is None:
        default_pg = _get_default_group()
        
        work = default_pg._get_backend(torch.device("npu")).barrier(opts=opts)
    else:
        work = group._get_backend(torch.device("npu")).barrier(opts=opts)

    if async_op:
        return work
    else:
        work.wait()


def apply_c10d_patch():
    torch.distributed.barrier = barrier
