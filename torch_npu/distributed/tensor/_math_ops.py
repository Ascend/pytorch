# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    register_op_strategy,
    normalize_dim
)
from torch.distributed.tensor._ops._math_ops import (
    _replicate_dims_start_at,
    _infer_reduce_dims_map,
    map_placements_after_reduction)
from torch.distributed.tensor._utils import normalize_to_torch_size
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding

npu = torch.ops.npu


@register_op_strategy(npu.npu_rms_norm.default)
def npu_rms_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    expected_args_len = 2
    (
        input_strategy,
        gamma_strategy,
    ) = op_schema.args_schema[:expected_args_len]

    normalized_shape = gamma_strategy.shape
    normalized_size = normalize_to_torch_size(normalized_shape)

    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)

    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )

        if gamma_strategy is not None:
            gamma_src_spec = gamma_strategy.strategies[idx].output_spec

            gamma_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(gamma_src_spec.placements),
                tensor_meta=gamma_src_spec.tensor_meta,
            )
            op_args_target_specs.append(gamma_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(gamma_strategy, gamma_target_spec)
            )

        # the output spec is the same as input spec
        output_target_spec = input_target_spec
        output_strategy.strategies.append(
            PlacementStrategy(
                output_specs=output_target_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy(npu.npu_rms_norm_backward.default)
def npu_rms_norm_backward_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    (
        grad_out_strategy,
        input_strategy,
        gamma_strategy,
        rstd_strategy,
    ) = op_schema.args_schema

    normalized_shape = gamma_strategy.shape
    normalized_size = normalize_to_torch_size(normalized_shape)
    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)
    outer_dims = list(range(axis))

    out_tuple_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        output_specs_list: list[Optional[DTensorSpec]] = []
        input_specs_list: list[DTensorSpec] = []
        redistribute_costs = []

        input_src_spec = input_placement_strategy.output_spec
        grad_out_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(grad_out_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(grad_out_strategy, grad_out_target_spec)
        )
        output_specs_list.append(grad_out_target_spec)

        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(input_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )

        if gamma_strategy is not None:
            gamma_src_spec = gamma_strategy.strategies[idx].output_spec
            input_specs_list.append(gamma_src_spec)
            redistribute_costs.append([0.0 for _ in gamma_strategy.strategies])
            # we may need to change to a pointwise rule over grad_out and
            # input, then apply a reduction.
            inp_placements = _replicate_dims_start_at(input_src_spec.placements, axis)
            reduce_dims_map = _infer_reduce_dims_map(
                outer_dims, input_src_spec.ndim, False
            )
            out_placements = map_placements_after_reduction(
                inp_placements, outer_dims, reduce_dims_map, "sum"
            )
            gamma_out_spec = DTensorSpec(
                mesh=mesh,
                placements=out_placements,
                tensor_meta=gamma_src_spec.tensor_meta,
            )
            output_specs_list.append(gamma_out_spec)
        else:
            output_specs_list.append(None)

        rstd_src_spec = rstd_strategy.strategies[idx].output_spec
        input_specs_list.append(rstd_src_spec)
        redistribute_costs.append([0.0 for _ in rstd_strategy.strategies])

        out_tuple_strategy.strategies.append(
            PlacementStrategy(
                output_specs=tuple(output_specs_list),
                input_specs=input_specs_list,
                redistribute_cost=redistribute_costs,
            )
        )

    return out_tuple_strategy


@register_op_strategy(npu.npu_add_rms_norm.default)
def npu_add_rms_norm_strategy(op_schema: OpSchema) -> OpStrategy:
    # func: npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor)
    mesh = op_schema.get_mesh_from_args(validate=False)
    expected_args_len = 3
    (
        x1_strategy,
        x2_strategy,
        gamma_strategy,
    ) = op_schema.args_schema[:expected_args_len]

    normalized_shape = gamma_strategy.shape
    normalized_size = normalize_to_torch_size(normalized_shape)

    # x1, x2 should have the same shape
    input_ndim = x1_strategy.ndim
    axis = input_ndim - len(normalized_size)

    output_strategy = OpStrategy([])
    for idx, x1_placement_strategy in enumerate(x1_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []

        # dims to calculate rms norm should be Replicate
        x1_src_spec = x1_placement_strategy.output_spec
        x1_target_placements = _replicate_dims_start_at(x1_src_spec.placements, axis)
        x1_target_spec = DTensorSpec(
            mesh=mesh,
            placements=x1_target_placements,
            tensor_meta=x1_src_spec.tensor_meta,
        )
        op_args_target_specs.append(x1_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(x1_strategy, x1_target_spec)
        )

        # x2 follows x1
        if x2_strategy is not None:
            x2_src_spec = x2_strategy.strategies[idx].output_spec
            x2_target_spec = DTensorSpec(
                mesh,
                placements=x1_target_placements,
                tensor_meta=x2_src_spec.tensor_meta,
            )
            op_args_target_specs.append(x2_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(x2_strategy, x2_target_spec)
            )

        if gamma_strategy is not None:
            gamma_src_spec = gamma_strategy.strategies[idx].output_spec
            gamma_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(gamma_src_spec.placements),
                tensor_meta=gamma_src_spec.tensor_meta,
            )
            op_args_target_specs.append(gamma_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(gamma_strategy, gamma_target_spec)
            )

        y_target_spec = x1_target_spec
        rstd_target_spec = DTensorSpec(
            mesh=mesh,
            placements=x1_target_placements[:axis],
            tensor_meta=x1_src_spec.tensor_meta,
        )
        x_target_spec = x1_target_spec
        output_target_spec = (y_target_spec, rstd_target_spec, x_target_spec)

        output_strategy.strategies.append(
            OpSpec(
                output_specs=output_target_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_sharding(npu.npu_rotary_mul.default)
def npu_rotary_mul_strategy(x, r1, r2, rotary_mode="half"):
    # func: npu_rotary_mul(Tensor self, Tensor r1, Tensor r2, str rotary_mode='half') -> Tensor
    acceptable_shardings = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate()], # output
        [Replicate(), Replicate(), Replicate(), None] # x, r1, r2, rotary_mode
    )
    acceptable_shardings.append(replicate_strategy)

    # sharding strategy
    # for any layout of x, the last dim always be D, which is not shardable
    for i in range(x.ndim - 1):
        # x, r1, r2 have the same layout, while B/N in r1/r2 can be 1, r1 and r2 have the same shape
        if r1.shape[i] == 1:
            sharding_strategy = (
                [Shard(i)],
                [Shard(i), Replicate(), Replicate(), None]
            )
        else:
            sharding_strategy = (
                [Shard(i)],
                [Shard(i), Shard(i), Shard(i), None]
            )
        acceptable_shardings.append(sharding_strategy)

    return acceptable_shardings


@register_sharding(npu.npu_rotary_mul_backward.default)
def npu_rotary_mul_backward_strategy(grad, x, r1, r2, rotary_mode='half'):
    # func: npu_rotary_mul_backward(Tensor grad, Tensor self, Tensor r1, Tensor r2, str rotary_mode='half')
    #                              -> (Tensor, Tensor, Tensor)
    acceptable_shardings = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate(), Replicate(), Replicate()], # dx, d_r1, d_r2
        [Replicate(), Replicate(), Replicate(), Replicate(), None] # grad, x, r1, r2, rotary_mode
    )
    acceptable_shardings.append(replicate_strategy)

    # sharding strategy
    # for any layout of x, the last dim always be D, which is not shardable
    for i in range(x.ndim - 1):
        # x, r1, r2 have the same layout, while B/N in r1, r2 can be 1, r1 and r2 have the same shape
        if r1.shape[i] == 1:
            sharding_strategy = (
                [Shard(i), Partial(), Partial()],
                [Shard(i), Shard(i), Replicate(), Replicate(), None]
            )
        else:
            sharding_strategy = (
                [Shard(i), Shard(i), Shard(i)],
                [Shard(i), Shard(i), Shard(i), Shard(i), None]
            )
        acceptable_shardings.append(sharding_strategy)

    return acceptable_shardings


@register_sharding(npu.npu_gather_backward.default)
def custom_npu_gather_backward_strategy(grad, input_shape, dim, index, sparse_grad=False):
    dim = normalize_dim(dim, len(input_shape))
    index_shape = index.shape
    grad_shape = grad.shape

    acceptable_sharding = []

    replicate_strategy = ([Replicate()], [Replicate(), None, None, Replicate(), None])
    acceptable_sharding.append(replicate_strategy)

    if len(input_shape) == len(index_shape) == len(grad_shape):
        for d in range(index.ndim):
            if d != dim and input_shape[d] == index_shape[d] == grad_shape[d]:
                sharding_strategy = (
                    [Shard(d)],    # output(grad_input)
                    [Shard(d),    # grad
                     None,    # input
                     None,    # dim
                     Shard(d),    # index
                     None]    # sparse_grad
                )
                acceptable_sharding.append(sharding_strategy)

    return acceptable_sharding


@register_sharding(npu.npu_swiglu.default)
def custom_npu_swiglu_strategy(x, dim=-1):
    dim = normalize_dim(dim, x.ndim)

    acceptable_sharding = []

    replicate_strategy = ([Replicate()], [Replicate(), None])
    acceptable_sharding.append(replicate_strategy)

    for i in range(x.ndim):
        if i != dim:
            sharding_strategy = (
                [Shard(i)],
                [Shard(i), None]
            )
            acceptable_sharding.append(sharding_strategy)

    return acceptable_sharding


@register_sharding(npu.npu_swiglu_backward.default)
def custom_npu_swiglu_backward_strategy(grad, x, dim=-1):
    dim = normalize_dim(dim, x.ndim)

    acceptable_sharding = []

    replicate_strategy = (
        [Replicate(), None],
        [Replicate(), Replicate(), None]
    )
    acceptable_sharding.append(replicate_strategy)

    for i in range(x.ndim):
        if i != dim:
            sharding_strategy = (
                [Shard(i), None],
                [Shard(i), Shard(i), None]
            )
            acceptable_sharding.append(sharding_strategy)

    return acceptable_sharding
