from typing import List

import torch
from torch.distributed._tensor import Partial, Replicate, Shard
from torch.distributed._tensor.experimental import register_sharding
from torch.distributed._tensor.placement_types import DTensorSpec

aten = torch.ops.aten
npu = torch.ops.npu


def _get_max_shardable_dim(tensor):
    shape = tensor.shape
    world_size = torch.distributed.get_world_size()
    divisible_dims = [
        (idx, dim) for idx, dim in enumerate(shape) if dim % world_size == 0
    ]
    if divisible_dims:
        idx, _ = max(divisible_dims, key=lambda x: x[1])
        return idx
    else:
        return -1


@register_sharding(aten.matmul.default)
def custom_matmul_sharding(
    tensor1: DTensorSpec,
    tensor2: DTensorSpec,
):
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    max_dim1_index = _get_max_shardable_dim(tensor1)
    max_dim2_index = _get_max_shardable_dim(tensor2)

    acceptable_shardings = []

    if max_dim1_index == -1 and max_dim2_index == -1:
        strategy = ([Replicate()], [Replicate(), Replicate()])
        acceptable_shardings.append(strategy)
        return acceptable_shardings
    elif max_dim1_index == -1:
        max_dim1_size = 0
        max_dim2_size = shape2[max_dim2_index]
    elif max_dim2_index == -1:
        max_dim1_size = shape1[max_dim1_index]
        max_dim2_size = 0
    else:
        max_dim1_size = shape1[max_dim1_index]
        max_dim2_size = shape2[max_dim2_index]

    max_size_in_1 = max_dim1_size >= max_dim2_size
    max_size_in_2 = max_dim1_size < max_dim2_size

    if len(shape1) == 1 and len(shape2) == 1:  # n@n=1
        strategy = ([Replicate()], [Replicate(), Replicate()])
    elif len(shape1) == 1:  # n@...nm=...m
        if max_size_in_1:
            strategy = ([Partial()], [Shard(max_dim1_index), Shard(len(shape2) - 2)])
        elif max_dim2_index == len(shape2) - 1:
            strategy = ([Shard(-1)], [Replicate(), Shard(max_dim2_index)])
        else:
            strategy = ([Shard(max_dim2_index)], [Replicate(), Shard(max_dim2_index)])
    elif len(shape2) == 1:  # ...nm@m=...n
        if max_size_in_1 and not max_dim1_index == len(shape1) - 1:
            strategy = ([Shard(max_dim1_index)], [Shard(max_dim1_index), Replicate()])
        else:
            strategy = ([Partial()], [Shard(max_dim1_index), Shard(0)])
    else:  # ...nm@...mk=...nk(braodcast)
        output_shape = torch.broadcast_shapes(shape1[:-2], shape2[:-2]) + (
            shape1[-2],
            shape2[-1],
        )
        if max_size_in_1 and not max_dim1_index == len(shape1) - 1:
            strategy = (
                [Shard(len(output_shape) - (len(shape1) - max_dim1_index))],
                [Shard(max_dim1_index), Replicate()],
            )
        elif max_size_in_2 and not max_dim1_index == len(shape2) - 2:
            strategy = (
                [Shard(len(output_shape) - (len(shape2) - max_dim2_index))],
                [Replicate(), Shard(max_dim2_index)],
            )
        else:
            strategy = ([Partial()], [Shard(len(shape1) - 1), Shard(len(shape2) - 2)])
    acceptable_shardings.append(strategy)
    return acceptable_shardings


@register_sharding(aten.matmul_backward.default)
def custom_matmul_backward_sharding(
    grad: DTensorSpec,
    self: DTensorSpec,
    other: DTensorSpec,
    mask: List[bool],
):
    acceptable_shardings = []
    grad_dim = len(grad.shape)
    self_dim = len(self.shape)
    other_dim = len(other.shape)
    if self_dim == 1 and other_dim == 1:
        strategy = (
            [Replicate(), Replicate()],
            [Replicate(), Replicate(), Replicate(), None],
        )
    elif (
        other_dim == 1
        and self_dim >= 2
        and self.shape[-2] % torch.distributed.get_world_size() == 0
    ):
        strategy = ([Shard(-2), Partial()], [Shard(-1), Shard(-2), Replicate(), None])
    elif (
        self_dim >= 1
        and other_dim >= 2
        and self.shape[-1] % torch.distributed.get_world_size() == 0
    ):
        strategy = ([Shard(-1), Shard(-2)], [Replicate(), Shard(-1), Shard(-2), None])
    else:
        strategy = (
            [Replicate(), Replicate()],
            [Replicate(), Replicate(), Replicate(), None],
        )

    acceptable_shardings.append(strategy)
    return acceptable_shardings


@register_sharding(npu.npu_dtype_cast_backward.default)
def custom_npu_dtype_cast_backward_sharding(
    grad: DTensorSpec,
    dtype: torch.dtype,
):
    max_dim_index = _get_max_shardable_dim(grad)
    if not max_dim_index == -1:
        strategy = ([Shard(max_dim_index)], [Shard(max_dim_index), None])
    else:
        strategy = ([Replicate()], [Replicate(), None])
    acceptable_shardings = []
    acceptable_shardings.append(strategy)
    return acceptable_shardings
