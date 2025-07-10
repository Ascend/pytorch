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
)
from torch.distributed.tensor._ops._math_ops import (
    _replicate_dims_start_at,
    _infer_reduce_dims_map,
    map_placements_after_reduction)
from torch.distributed.tensor._utils import normalize_to_torch_size

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
