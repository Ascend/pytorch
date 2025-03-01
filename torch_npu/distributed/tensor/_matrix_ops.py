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
            output_shape = shape2[:-2] + (shape2[-1],)
            strategy = (
                [Shard(len(output_shape) - 1)],
                [Replicate(), Shard(max_dim2_index)],
            )
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
        strategy = (
            [Shard(self_dim - 2), Partial()],
            [Shard(grad_dim - 1), Shard(self_dim - 2), Replicate(), None],
        )
    elif (
        self_dim >= 1
        and other_dim >= 2
        and self.shape[-1] % torch.distributed.get_world_size() == 0
    ):
        strategy = (
            [Shard(self_dim - 1), Shard(other_dim - 2)],
            [Replicate(), Shard(self_dim - 1), Shard(other_dim - 2), None],
        )
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


@register_sharding(npu.npu_fusion_attention.default)
# pylint:disable=huawei-too-many-arguments
def custom_npu_fusion_attention_sharding(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                         atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
                                         inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None,
                                         sparse_mode=0, gen_mask_parallel=True, sync=False):
    acceptable_shardings = []

    # add all replicate strategy
    replcate_strategy = (
        [
            Replicate(),  # Tensor attention_score
            Replicate(),  # Tensor softmax_max
            Replicate(),  # Tensor softmax_sum
            Replicate(),  # Tensor softmax_out
            None,  # int seed
            None,  # int offset
            None  # int numels
        ],
        [
            Replicate(),  # Tensor query
            Replicate(),  # Tensor key
            Replicate(),  # Tensor value
            None,  # int head_num
            None,  # str input_layout
            None if pse is None else Replicate(),  # Tensor? pse
            None if padding_mask is None else Replicate(),  # Tensor? padding_mask
            None if atten_mask is None else Replicate(),  # Tensor? atten_mask
            None  # other
        ]
    )

    # add sharding strategy
    for strategy_index, default_sharding in enumerate(query.placements):
        pse_sharding = None if pse is None else pse.placements[strategy_index]
        padding_mask_sharding = None if padding_mask is None else padding_mask.placements[strategy_index]
        atten_mask_sharding = None if atten_mask is None else atten_mask.placements[strategy_index]

        sharding_strategy = (
            [
                default_sharding,  # Tensor attention_score
                default_sharding,  # Tensor softmax_max
                default_sharding,  # Tensor softmax_sum
                default_sharding,  # Tensor softmax_out
                None,  # int seed
                None,  # int offset
                None  # int numels
            ],
            [
                default_sharding,  # Tensor query
                default_sharding,  # Tensor key
                default_sharding,  # Tensor value
                None,  # int head_num
                None,  # str input_layout
                pse_sharding,  # Tensor? pse
                padding_mask_sharding,  # Tensor? padding_mask
                atten_mask_sharding,  # Tensor? atten_mask
                None  # other
            ]
        )

        acceptable_shardings.append(sharding_strategy)

    acceptable_shardings.append(replcate_strategy)

    return acceptable_shardings


@register_sharding(npu.npu_fusion_attention_grad.default)
# pylint:disable=huawei-too-many-arguments
def custom_npu_fusion_attention_grad_sharding(query, key, value, dy, head_num, input_layout, *, pse=None,
                                              padding_mask=None, atten_mask=None, softmax_max=None, softmax_sum=None,
                                              softmax_in=None, attention_in=None, scale_value=1.0, keep_prob=1.0,
                                              pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=0,
                                              offset=0, numels=0, prefix=None, actual_seq_qlen=None,
                                              actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    acceptable_shardings = []

    # add all replicate strategy
    replcate_strategy = (
        [
            Replicate(),  # Tensor grad_query
            Replicate(),  # Tensor grad_key
            Replicate(),  # Tensor grad_value
            Replicate(),  # Tensor grad_dy
        ],
        [
            Replicate(),  # Tensor query
            Replicate(),  # Tensor key
            Replicate(),  # Tensor value
            Replicate(),  # Tensor dy
            None,  # int head_num
            None,  # str input_layout
            None if pse is None else Replicate(),  # Tensor? pse
            None if padding_mask is None else Replicate(),  # Tensor? padding_mask
            None if atten_mask is None else Replicate(),  # Tensor? atten_mask
            None if softmax_max is None else Replicate(),  # Tensor? softmax_max
            None if softmax_sum is None else Replicate(),  # Tensor? softmax_sum
            None if softmax_in is None else Replicate(),  # Tensor? softmax_in
            None if attention_in is None else Replicate(),  # Tensor? attention_in
            None  # other
        ]
    )
    acceptable_shardings.append(replcate_strategy)

    # add sharding strategy
    for strategy_index, default_sharding in enumerate(query.placements):
        pse_sharding = None if pse is None else pse.placements[strategy_index]
        padding_mask_sharding = None if padding_mask is None else padding_mask.placements[strategy_index]
        atten_mask_sharding = None if atten_mask is None else atten_mask.placements[strategy_index]
        asoftmax_max_sharding = None if softmax_max is None else softmax_max.placements[strategy_index]
        softmax_sum_sharding = None if softmax_sum is None else softmax_sum.placements[strategy_index]
        softmax_in_sharding = None if softmax_in is None else softmax_in.placements[strategy_index]
        attention_in_sharding = None if attention_in is None else attention_in.placements[strategy_index]

        sharding_strategy = (
            [
                default_sharding,  # Tensor grad_query
                default_sharding,  # Tensor grad_key
                default_sharding,  # Tensor grad_value
                default_sharding,  # Tensor grad_dy
            ],
            [
                default_sharding,  # Tensor query
                default_sharding,  # Tensor key
                default_sharding,  # Tensor value
                default_sharding,  # Tensor dy
                None,  # int head_num
                None,  # str input_layout
                pse_sharding,  # Tensor? pse
                padding_mask_sharding,  # Tensor? padding_mask
                atten_mask_sharding,  # Tensor? atten_mask
                asoftmax_max_sharding,  # Tensor? softmax_max
                softmax_sum_sharding,  # Tensor? softmax_sum
                softmax_in_sharding,  # Tensor? softmax_sum
                attention_in_sharding,  # Tensor? attention_in
                None  # other
            ]
        )
        acceptable_shardings.append(sharding_strategy)

    return acceptable_shardings
