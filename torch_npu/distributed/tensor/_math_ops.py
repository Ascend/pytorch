# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import Optional, cast

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    OpSpec,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    register_op_strategy,
    normalize_dim,
    expand_to_full_mesh_op_strategy,
)
from torch.distributed.tensor._ops._math_ops import (
    _replicate_dims_start_at,
    _infer_reduce_dims_map,
    map_placements_after_reduction,
)
from torch.distributed.tensor._utils import normalize_to_torch_size
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding

npu = torch.ops.npu
aten = torch.ops.aten


@register_op_strategy(npu.npu_rms_norm.default)
def npu_rms_norm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=False)
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
            OpSpec(
                output_specs=output_target_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy(npu.npu_rms_norm_backward.default)
def npu_rms_norm_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=False)
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
            OpSpec(
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


@register_sharding(npu.npu_conv2d.default)
def custom_npu_conv2d_strategy(x, weight, bias, stride, padding, dilation, groups):
    acceptable_shardings = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate() # output
        ],
        [
            Replicate(),                           # x
            Replicate(),                           # weight
            None if bias is None else Replicate(), # bias
            None, None, None, None                 # others
        ]
    )
    acceptable_shardings.append(replicate_strategy)
    
    # x layout: (N, Ci, Hi, Wi)
    # weight layout: (Co, Ci/groups, Hk, Wk)
    # bias layout: (Co)
    # output layout: (N, Co, Ho, Wo)
    # dp sharding strategy
    N_sharding_strategy = (
        [Shard(0)],
        [Shard(0), Replicate(), None if bias is None else Replicate(), None, None, None, None]
    )
    acceptable_shardings.append(N_sharding_strategy)

    # tp sharding strategy when groups == 1
    if groups == 1:
        Co_sharding_strategy = (
            [Shard(1)],
            [Replicate(), Shard(0), None if bias is None else Shard(0), None, None, None, None]
        )
        acceptable_shardings.append(Co_sharding_strategy)

    return acceptable_shardings


@register_sharding(npu.npu_conv2d_backward.default)
def custom_npu_conv2d_backward_strategy(x, grad_output, weight, stride, padding, dilation, groups, output_mask):
    grad_output_placement = grad_output.placements[0]
    acceptable_shardings = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate(),                            # grad_x
            Replicate(),                            # grad_weight
            Replicate() if output_mask[2] else None # grad_bias
        ],
        [
            Replicate(),                            # x
            Replicate(),                            # grad_output
            Replicate(),                            # weight
            None, None, None, None, None            # others
        ]
    )
    acceptable_shardings.append(replicate_strategy)

    # Sharding by batch dimension (N-dimensional, 0-dimensional) - Input/output is sharded by N, with weights fully replicated.
    if (isinstance(grad_output_placement, Shard) and grad_output_placement.dim == 0):
        N_sharding_strategy = (
            [Shard(0), Partial(), Partial() if output_mask[2] else None],
            [Shard(0), Shard(0), Replicate(), None, None, None, None, None]
        )
        acceptable_shardings.append(N_sharding_strategy)

    # Sharding by output channel dimension (1D) - Weights/output are sharded by C_out, and inputs are fully replicated.
    if (isinstance(grad_output_placement, Shard) and grad_output_placement.dim == 1) and (groups == 1):
        Co_sharding_strategy = (
            [Partial(), Shard(0), Shard(0) if output_mask[2] else None],
            [Replicate(), Shard(1), Shard(0), None, None, None, None, None]
        )
        acceptable_shardings.append(Co_sharding_strategy)

    return acceptable_shardings


@register_sharding(npu.npu_grouped_matmul_add_.default)
def custom_grouped_matmul_add__strategy(y, x, weight, group_list, transpose_x=True, transpose_weight=False, group_type=2):
    acceptable_shardings = []
    y_placement = y.placements[0]

    if (isinstance(y_placement, Shard) and y_placement.dim == 0):
        raise ValueError(
            "The 0th dimension of the output matrix cannot be split."
        )
    # all replicate strategy
    replicate_strategy = (
        [
            Replicate()  # y
        ],
        [
            Replicate(), # y
            Replicate(), # x
            Replicate(), # weight
            Replicate(), # group_list
            None, None, None
        ]
    )
    acceptable_shardings.append(replicate_strategy)

    D_shard_strategy = (
        [Shard(1)], # y
        [Shard(1), Replicate(), Shard(1), Replicate(), None, None, None] # y, x, weight, group_list
    )
    acceptable_shardings.append(D_shard_strategy)

    return acceptable_shardings


def is_tensor_evenly_shardable(shape, spec):
    """Check if the shape is evenly shardable according to the spec."""
    # verify parameter validity    
    if not isinstance(spec, DTensorSpec):
        raise TypeError(
            f"Expected 'spec' to be DTensorSpec instance, got {type(spec).__name__} instead."
        )
    if len(shape) == 0:
        raise ValueError("'shape' must have at least 1 dimension (empty shape is invalid).")
    if len(spec.placements) == 0:
        raise ValueError("'spec.placements' cannot be empty (must have at least one placement).")
    
    # number of shards in each tensor dimension
    shards_map = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            shards_map[shard_dim] *= spec.mesh.size(i)

    for i, dim_size in enumerate(shape):
        if shards_map[i] > 1 and (dim_size % shards_map[i] != 0):
            return False

    return True


@register_op_strategy(npu.npu_cross_entropy_loss.default, schema_info=RuntimeSchemaInfo(3))
def custom_cross_entropy_loss_sharding(op_schema: OpSchema):
    single_mesh_dim_strategies = []

    args_schema = op_schema.args_schema
    
    input_strategy = args_schema[0] if len(args_schema) > 0 else None
    target_strategy = args_schema[1] if len(args_schema) > 1 else None
    weight_strategy = args_schema[2] if len(args_schema) > 2 else None
    reduction = args_schema[3] if len(args_schema) > 3 else 'mean'

    mesh = input_strategy.mesh
    
    all_replicate: PlacementList = [
        Replicate(), # loss
        Replicate(), # log_prob
        Replicate(), # zloss
        Replicate(), # lse_for_zloss
        Replicate(), # x
        Replicate()  # target
    ]
    if weight_strategy is not None:
        all_replicate.append(Replicate()) # weight
    single_mesh_dim_strategies.append(all_replicate)

    if reduction == 'none':
        N_replicate_strategy = [Shard(0), Shard(0), Replicate(), Replicate(), Shard(0), Shard(0)]
        if weight_strategy is not None:
            N_replicate_strategy.append(Replicate())
        single_mesh_dim_strategies.append(N_replicate_strategy)
    elif reduction == 'mean':
        target_shape = target_strategy.strategies[0].output_spec.shape
        target_spec = target_strategy.strategies[0].output_spec
        if weight_strategy is None and is_tensor_evenly_shardable(target_shape, target_spec):
            N_replicate_strategy = [Partial("avg"), Shard(0), Replicate(), Replicate(), Shard(0), Shard(0)]
            single_mesh_dim_strategies.append(N_replicate_strategy)
    else:
        N_replicate_strategy = [Partial("sum"), Shard(0), Replicate(), Replicate(), Shard(0), Shard(0)]
        if weight_strategy is not None:
            N_replicate_strategy.append(Replicate())
        single_mesh_dim_strategies.append(N_replicate_strategy)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=4
    )


@register_op_strategy(npu.npu_cross_entropy_loss_backward.default, schema_info=RuntimeSchemaInfo(6))
def custom_cross_entropy_loss_backward_strategy(op_schema: OpSchema):
    single_mesh_dim_strategies = []
    args_schema = op_schema.args_schema
    grad_loss_strategy = args_schema[0] if len(args_schema) > 0 else None
    weight_strategy = args_schema[3] if len(args_schema) > 3 else None
    lse_for_zloss_strategy = args_schema[5] if len(args_schema) > 5 else None
    reduction = args_schema[6] if len(args_schema) > 6 else 'mean'

    mesh = grad_loss_strategy.mesh

    all_replicate: PlacementList = [
        Replicate(),
        Replicate(),
        Replicate(),
        Replicate()
    ]
    if weight_strategy is not None:
        all_replicate.append(Replicate())
    if lse_for_zloss_strategy is not None:
        all_replicate.append(Replicate())
    single_mesh_dim_strategies.append(all_replicate)

    if reduction == 'none':
        N_replicate_strategy = [Shard(0), Shard(0), Shard(0), Shard(0)]
        if weight_strategy is not None:
            N_replicate_strategy.append(Replicate())
        if lse_for_zloss_strategy is not None:
            N_replicate_strategy.append(Replicate())
        single_mesh_dim_strategies.append(N_replicate_strategy)
    elif reduction == 'sum':
        N_replicate_strategy = [Shard(0), Replicate(), Shard(0), Shard(0)]
        if weight_strategy is not None:
            N_replicate_strategy.append(Replicate())
        if lse_for_zloss_strategy is not None:
            N_replicate_strategy.append(Replicate())
        single_mesh_dim_strategies.append(N_replicate_strategy)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


@register_sharding(aten.repeat_interleave.self_int)
def custom_npu_repeat_interleave_self_int_strategy(x, repeat, dim=None, output_size=None):
    acceptable_shardings = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate()      # output
        ],
        [
            Replicate(),     # x
            None, None, None # others
        ]
    )
    acceptable_shardings.append(replicate_strategy)

    if not is_tensor_evenly_shardable(x.shape, x):
        if dim is not None:
            for i in range(x.ndim):
                if i != dim:
                    sharding_strategy = (
                        [Shard(i)],
                        [Shard(i), None, None, None]
                    )
                acceptable_shardings.append(sharding_strategy)
    else:
        if dim is None:
            sharding_strategy = (
                [Shard(0)],
                [Shard(0), None, None, None]
            )
            acceptable_shardings.append(sharding_strategy)
        else:
            for i in range(x.ndim):
                sharding_strategy = (
                    [Shard(i)],
                    [Shard(i), None, None, None]
                )
                acceptable_shardings.append(sharding_strategy)

    return acceptable_shardings


@register_sharding(npu.repeat_interleave_backward_int.default)
def custom_npu_repeat_interleave_backward_int_strategy(grad, x, repeats, dim=None):
    acceptable_shardings = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate()      # grad_x
        ],
        [
            Replicate(),     # grad
            Replicate(),     # x
            None, None       # others
        ]
    )
    acceptable_shardings.append(replicate_strategy)

    if dim is not None and is_tensor_evenly_shardable(x.shape, x):
        for i in range(x.ndim):
            grad_placement = grad.placements[0]
            if (isinstance(grad_placement, Shard)
                and dim == grad_placement.dim
                and not is_tensor_evenly_shardable(grad.shape, grad)
            ):
                continue
            else:
                sharding_strategy = (
                    [Shard(i)],
                    [Shard(i), Shard(i), None, None]
                )
                acceptable_shardings.append(sharding_strategy)

    return acceptable_shardings
