__all__ = []

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


def _allgather_base_backward_hccl(ctx, grad_output):
    """
    Backward function for _AllGatherBase that supports HCCL backend.
    
    Original PyTorch implementation only supports NCCL backend.
    This version adds HCCL support for NPU devices.
    """
    if dist.get_backend(group=ctx.group) in (dist.Backend.NCCL, dist.Backend.HCCL):
        world_size = dist.get_world_size(group=ctx.group)
        out_size = list(grad_output.size())
        if out_size[0] % world_size != 0:
            raise RuntimeError(
                f"Tensor with dimensions: {out_size} does "
                f"not have first dimension divisible by world_size: {world_size}"
            )
        out_size[0] = out_size[0] // world_size
        gx = torch.empty(
            out_size, device=grad_output.device, dtype=grad_output.dtype
        )
        dist._reduce_scatter_base(gx, grad_output, ReduceOp.SUM, ctx.group)
    else:
        raise RuntimeError("Backend not supported!")
    return (None, gx, None)
