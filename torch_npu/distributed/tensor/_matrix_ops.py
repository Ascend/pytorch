from typing import cast, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed._tensor.experimental import register_sharding
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._ops.utils import register_op_strategy, expand_to_full_mesh_op_strategy
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor._op_schema import (
    _is_out_variant_op,
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementStrategy,
    RuntimeSchemaInfo,
    TupleStrategy
)

import torch_npu

try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree

from ._common import (
    get_redistributed_local_args,
    get_redistributed_local_kwargs,
    get_empty_local_results
)

aten = torch.ops.aten
npu = torch.ops.npu


def _get_max_shardable_dim(tensor):
    shape = tensor.shape
    world_size = torch.distributed.get_world_size()
    divisible_dims = [(idx, dim) for idx, dim in enumerate(shape) if dim % world_size == 0]
    if divisible_dims:
        idx, _ = max(divisible_dims, key=lambda x: x[1])
        return idx
    else:
        return -1


def _handle_tensor_list_in_kwargs(kwargs: Dict[str, object], op_info: OpInfo) -> None:
    for key, value in kwargs.items():
        if isinstance(value, list) and all(isinstance(e, DTensor) for e in value):
            new_schema = []
            new_local_tensors = []
            for dtensor in value:
                new_schema.append(dtensor._spec)
                new_local_tensors.append(dtensor._local_tensor)
            op_info.schema.kwargs_schema[key] = tuple(new_schema) # list is not hashable for cache
            op_info.local_kwargs[key] = new_local_tensors


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


@register_op_strategy(
    npu.npu_grouped_matmul.default,
    schema_info=RuntimeSchemaInfo(
        static_kwargkey=["bias", "scale", "offset", "antiquant_scale", "antiquant_offset", "per_token_scale",
                         "group_list", "activation_input", "activation_quant_scale", "activation_quant_offset"],
        needs_pytree=True
    )
)
@register_op_strategy(
    npu.npu_grouped_matmul.List,
    schema_info=RuntimeSchemaInfo(
        static_kwargkey=["bias", "scale", "offset", "antiquant_scale", "antiquant_offset", "per_token_scale",
                         "activation_input", "activation_quant_scale", "activation_quant_offset"],
        needs_pytree=True
    )
)
def npu_grouped_matmul_strategy(op_schema: OpSchema) -> OpStrategy:
    # npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor[]? bias=None, Tensor[]? scale=None,
    #                    Tensor[]? offset=None, Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None,
    #                    Tensor[]? per_token_scale=None, Tensor? group_list=None, Tensor[]? activation_input=None,
    #                    Tensor[]? activation_quant_scale=None, Tensor[]? activation_quant_offset=None,
    #                    int? split_item=0, int? group_type=None, int? group_list_type=0, int? act_type=0,
    #                    int[]? tuning_config=None, int? output_dtype=None, int? x_dtype=None, int? weight_dtype=None,
    #                    int? scale_dtype=None, int? per_token_scale_dtype=None) -> Tensor[]
    if op_schema.schema_info is None:
        op_schema.schema_info = RuntimeSchemaInfo(needs_pytree=True) # to flatten tensor list in arguments
    x_src_strategy: TupleStrategy = op_schema.args_schema[0]
    x_num = len(x_src_strategy.childs)
    weight_src_strategy: TupleStrategy = op_schema.args_schema[1]
    weight_num = len(weight_src_strategy.childs)
    bias_src_strategy: Optional[Union[TupleStrategy, list]] = op_schema.kwargs_schema.get("bias", [])
    bias_num = len(bias_src_strategy.childs) if isinstance(bias_src_strategy, TupleStrategy) else len(bias_src_strategy)
    group_list_num = 1 if (
        op_schema.op == npu.npu_grouped_matmul.default and
        op_schema.kwargs_schema.get("group_list", None) is not None
    ) else 0
    split_item = op_schema.kwargs_schema.get("split_item", 0)
    y_num = weight_num if split_item in (0, 1) else 1 # 0/1: multiple outputs, 2/3: single output

    strategies = []

    all_replicate_strategy = [Replicate()] * y_num
    all_replicate_strategy.extend([Replicate()] * (len(op_schema.args_strategy) + len(op_schema.kwargs_strategy)))
    strategies.append(all_replicate_strategy)

    unsupported_arguments = [
        "scale", "offset", "antiquant_scale", "antiquant_offset", "per_token_scale", # quant
        "activation_input", "activation_quant_scale", "activation_quant_offset",     # reserved, unused now
    ]
    for key in unsupported_arguments:
        schema = op_schema.kwargs_schema.get(key, None)
        if schema is not None and isinstance(schema, TupleStrategy) and len(schema.childs) > 0:
            full_mesh_strategies = expand_to_full_mesh_op_strategy(
                op_schema.get_mesh_from_args(), op_schema, strategies, input_index=y_num
            )
            if y_num == 1:
                for strategy in full_mesh_strategies.strategies:
                    strategy.output_specs = [strategy.output_specs]
            return full_mesh_strategies

    if bias_num == 0: # if y is partial and bias exists, the bias will be added multiple times to the full tensor
        replicate_partial_strategy = [Partial()] * y_num
        replicate_partial_strategy.extend([Replicate()] * x_num)
        replicate_partial_strategy.extend([Partial()] * weight_num)
        replicate_partial_strategy.extend([Replicate()] * group_list_num)
        strategies.append(replicate_partial_strategy)

        partial_replicate_strategy = [Partial()] * y_num
        partial_replicate_strategy.extend([Partial()] * x_num)
        partial_replicate_strategy.extend([Replicate()] * weight_num)
        partial_replicate_strategy.extend([Replicate()] * group_list_num)
        strategies.append(partial_replicate_strategy)

    group_type = op_schema.kwargs_schema.get("group_type", None)
    if group_type is not None and group_type > 0:
        raise NotImplementedError(f"npu_grouped_matmul does not support group_type={group_type} now.")

    if x_num > 1 and weight_num > 1 and y_num > 1: # x_num, weight_num, y_num are equal
        pair_strategies = []
        # x: 2-6D, weight: 2D, weight: 1D (equals to weight.shape[1])
        # shard x
        x_ndim = x_src_strategy.childs[0].ndim
        for i in range(x_ndim - 1):
            pair_strategies.append([Shard(i), Shard(i), Replicate(), Replicate()]) # y, x, weight, bias
        # shard weight
        pair_strategies.append([Shard(x_ndim - 1), Replicate(), Shard(1), Shard(0)])
        # shard contracting dim
        if bias_num == 0:
            pair_strategies.append([Partial(), Shard(x_ndim - 1), Shard(0), None])
        # suppose that all pairs have the same shape and apply the same strategy
        for (y_spec, x_spec, weight_spec, bias_spec) in pair_strategies:
            strategy = [y_spec] * y_num
            strategy.extend([x_spec] * x_num)
            strategy.extend([weight_spec] * weight_num)
            strategy.extend([bias_spec] * bias_num)
            strategy.extend([Replicate()] * group_list_num)
            strategies.append(strategy)
    elif x_num == 1 and weight_num == 1 and y_num == 1: # npu_grouped_matmul.default only
        # x: 2D, weight: 3D, bias: 2D, y: 2D, for each pair, define shape x: (m, k), weight: (k, n)
        if bias_num == 0:
            k_shard_strategy = [Partial(), Shard(1), Shard(1)]
            k_shard_strategy.extend([Replicate()] * group_list_num)
            strategies.append(k_shard_strategy)
        n_shard_strategy = [Shard(1), Replicate(), Shard(2)]
        n_shard_strategy.extend([Shard(1)] * bias_num)
        n_shard_strategy.extend([Replicate()] * group_list_num)
        strategies.append(n_shard_strategy)
    elif weight_num > 1: # x1wNy1, xNwNy1, x1wNyN
        # x: 2D, weight: 2D, bias: 1D, y: 2D
        if bias_num == 0:
            k_shard_strategy = [Partial()] * y_num
            k_shard_strategy.extend([Shard(1)] * x_num)
            k_shard_strategy.extend([Shard(0)] * weight_num)
            k_shard_strategy.extend([Replicate()] * group_list_num)
            strategies.append(k_shard_strategy)
        n_shard_strategy = [Shard(1)] * y_num
        n_shard_strategy.extend([Replicate()] * x_num)
        n_shard_strategy.extend([Shard(1)] * weight_num)
        n_shard_strategy.extend([Shard(0)] * bias_num)
        n_shard_strategy.extend([Replicate()] * group_list_num)
        strategies.append(n_shard_strategy)

    full_mesh_strategies = expand_to_full_mesh_op_strategy(op_schema.get_mesh_from_args(), op_schema, strategies,
                                                           input_index=y_num)
    # output meta of npu_grouped_matmul is list, need convert output_spec here
    if y_num == 1:
        for strategy in full_mesh_strategies.strategies:
            strategy.output_specs = [strategy.output_specs]
    return full_mesh_strategies


def _infer_npu_grouped_matmul_kwargs(
    op_schema: OpSchema,
    output_sharding: OutputSharding
) -> Dict[str, DTensorSpec]:
    output_spec = output_sharding.output_spec[0]
    kwargs_spec = {}
    for key, spec in op_schema.kwargs_schema.items():
        is_tensor_or_tenor_list_like = (
            isinstance(spec, DTensorSpec) or
            (isinstance(spec, (list, tuple)) and len(spec) > 0 and isinstance(spec[0], DTensorSpec))
        )
        if not is_tensor_or_tenor_list_like:
            kwargs_spec[key] = spec
            continue

        if key == 'group_list': # tensor
            target_placement = [Replicate() for _ in output_spec.placements]
            kwargs_spec[key] = DTensorSpec(mesh=spec.mesh, placements=target_placement, tensor_meta=spec.tensor_meta)
            continue

        # tensor list
        if key == 'bias':
            target_placement = [
                Shard(0) if placement == Shard(output_spec.ndim - 1) else Replicate()
                for placement in output_spec.placements
            ]
        else: # unsupported sharding keys
            target_placement = [Replicate() for _ in output_spec.placements]
        kwargs_spec[key] = [DTensorSpec(mesh=e.mesh, placements=target_placement, tensor_meta=e.tensor_meta) for e in spec]

    return kwargs_spec


def _npu_grouped_matmul_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    # since upwrap_to_op_info does not process List[DTensor] in kwargs, we need to handle it here
    _handle_tensor_list_in_kwargs(kwargs, op_info)

    # return type of npu_grouped_matmul is tensor list, which caused output_spec to be None after propagation, and
    # v2.9.0 fixed it. We set return_type_tensor to True to avoid patching the entire propagate_op_sharding_non_cached
    # function in previous versions.
    def _return_type_tensor():
        return True

    op_info.schema.return_type_tensor = _return_type_tensor

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding

    mesh = op_info.compute_mesh
    participating = mesh.get_coordinate() is not None
    if participating:
        # computation that happens in the current rank of the mesh, normal case
        local_args = get_redistributed_local_args(op_info, output_sharding)
        local_kwargs = get_redistributed_local_kwargs(_infer_npu_grouped_matmul_kwargs, op_info, output_sharding)
        local_results = op_call(*local_args, **local_kwargs)
    else:
        # For a non-participating device (happens on rank that does not belong to the device mesh),
        # return empty tensor(s) with correct dtype.
        local_results = get_empty_local_results(op_info, output_sharding)

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


@register_sharding(npu.npu_all_gather_base_mm.default)
def npu_all_gather_base_mm_strategy(x1, x2, hcom, world_size, bias=None, x1_scale=None, x2_scale=None, gather_index=0,
                                    gather_output=True, comm_turn=0, output_dtype=None, comm_mode=None):
    # npu_all_gather_base_mm(Tensor input, Tensor x2, str hcom, int world_size, *, Tensor? bias=None,
    #                        Tensor? x1_scale=None, Tensor? x2_scale=None, int gather_index=0, bool gather_output=True,
    #                        int comm_turn=0, ScalarType? output_dtype=None, str? comm_mode=None) -> (Tensor, Tensor)
    # op only support gather_index=0(i.e. allgather x1) now
    if gather_index != 0:
        raise NotImplementedError(f"npu_all_gather_base_mm only support gather_index=0 now, but got {gather_index}.")

    # formula: output = allgather(x1)@x2 + bias
    # for all gather, x1: S(0) -> R
    # for mm, when x1 is R, possible strategies are (R, R) -> R and (R, S(1)) -> S(1)
    # therefore, strategies of all_gather_base_mm are:
    # 1. (S(0), R) -> R, R
    # 2. (S(0), S(1)) -> S(1), R
    strategies = []
    sharding_strategy_S0R = (
        [
            Replicate(), # output
            Replicate()  # gather_out
        ],
        [
            Shard(0),    # x1
            Replicate(), # x2
            None,        # hcom
            None,        # world_size
            None if bias is None else Replicate(),     # bias, global shape(n * world_size,)
            None if x1_scale is None else Shard(0),    # x1_scale follow x1, global shape(m * world_size, 1)
            None if x2_scale is None else Replicate(), # x2_scale follow x2, global shape(1, n * world_size)
            None, None, None, None, None # gather_index, gather_output, comm_turn, output_dtype, comm_mode
        ]
    )
    strategies.append(sharding_strategy_S0R)

    sharding_strategy_S0S1 = (
        [
            Shard(1),   # output
            Replicate() # gather_out
        ],
        [
            Shard(0), # x1
            Shard(1), # x2
            None,     # hcom
            None,     # world_size
            None if bias is None else Shard(0),     # bias, global shape(n * world_size,)
            None if x1_scale is None else Shard(0), # x1_scale follow x1, global shape(m * world_size, 1)
            None if x2_scale is None else Shard(1), # x2_scale follow x2, global shape(1, n * world_size)
            None, None, None, None, None # gather_index, gather_output, comm_turn, output_dtype, comm_mode
        ]
    )
    strategies.append(sharding_strategy_S0S1)

    return strategies


def _infer_npu_all_gather_base_mm_kwargs(
    op_schema: OpSchema,
    output_sharding: OutputSharding
) -> Dict[str, DTensorSpec]:
    output_spec = output_sharding.output_spec[0]
    kwargs_spec = {}
    for key, spec in op_schema.kwargs_schema.items():
        if not isinstance(spec, DTensorSpec):
            kwargs_spec[key] = spec
            continue

        target_placement = []
        for placement in output_spec.placements:
            if placement == Replicate():
                if key == 'x1_scale':
                    target_placement.append(Shard(0))
                else: # bias, x2_scale
                    target_placement.append(Replicate())
            elif placement == Shard(1):
                if key == 'x2_scale':
                    target_placement.append(Shard(1))
                else: # bias, x1_scale
                    target_placement.append(Shard(0))
            else:
                raise ValueError(
                    f"Unexpected output placement {placement} for npu_all_gather_base_mm."
                )
        kwargs_spec[key] = DTensorSpec(mesh=spec.mesh, placements=target_placement, tensor_meta=spec.tensor_meta)

    return kwargs_spec


@register_sharding(npu.npu_mm_reduce_scatter_base.default)
def npu_mm_reduce_scatter_base_strategy(x1, x2, hcom, world_size, reduce_op='sum', bias=None, x1_scale=None,
                                        x2_scale=None, comm_turn=0, output_dtype=None, comm_mode=None):
    # npu_mm_reduce_scatter_base(Tensor self, Tensor x2, str hcom, int world_size, *, str reduce_op='sum',
    #                            Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, int comm_turn=0,
    #                            ScalarType? output_dtype=None, str? comm_mode=None) -> Tensor
    # op only support reduce_op='sum' now
    if reduce_op != 'sum':
        raise NotImplementedError(f"npu_mm_reduce_scatter_base only support reduce_op='sum' now, but got {reduce_op}.")

    # formula: output = reducescatter(x1@x2 + bias)
    # for reduce_scatter, local_output: P -> S(0)
    # for mm, when output is P, possible strategies is (S(1), S(0)) -> P
    # therefore, strategy of mm_reduce_scatter_base is: (S(1), S(0)) -> S(0)
    strategies = []
    sharding_strategy_S1S0 = (
        [
            Shard(0)  # output
        ],
        [
            Shard(1), # x1
            Shard(0), # x2
            None,     # hcom
            None,     # world_size
            None,     # reduce_op
            None if bias is None else Shard(0),     # bias, global shape(n * world_size,)
            None if x1_scale is None else Shard(1), # x1_scale follow x1, global shape(m, world_size)
            None if x2_scale is None else Shard(0), # x2_scale follow x2, global shape(world_size, n)
            None, None, None # comm_turn, output_dtype, comm_mode
        ]
    )
    strategies.append(sharding_strategy_S1S0)

    return strategies


def _infer_npu_mm_reduce_scatter_base_kwargs(
    op_schema: OpSchema,
    output_sharding: OutputSharding
) -> Dict[str, DTensorSpec]:
    output_spec = output_sharding.output_spec
    kwargs_spec = {}
    for key, spec in op_schema.kwargs_schema.items():
        if not isinstance(spec, DTensorSpec):
            kwargs_spec[key] = spec
            continue

        target_placement = []
        for placement in output_spec.placements:
            if placement == Shard(0):
                if key == 'x1_scale':
                    target_placement.append(Shard(1))
                else: # bias, x2_scale
                    target_placement.append(Shard(0))
            else:
                raise ValueError(
                    f"Unexpected output placement {placement} for npu_mm_reduce_scatter_base."
                )
        kwargs_spec[key] = DTensorSpec(mesh=spec.mesh, placements=target_placement, tensor_meta=spec.tensor_meta)

    return kwargs_spec


def npu_comm_mm_fusion_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding

    # For all_gather_base_mm, output shape = (local_x1.shape[0] * world_size,local_x2.shape[1]),
    # but in ShardingPropagator.propagate, output meta is calculated based on dtensor input meta.
    # Since possible target input placement is Shard, i,e local input shape = meta shape / world_size,
    # we need correct output meta by dividing world_size here. For mm_reduce_scatter_base, similar process.
    def get_output_meta(tensor_meta, dim, world_size):
        if world_size == 0:
            return tensor_meta
        new_shape = list(tensor_meta.shape)
        if op_call == npu.npu_all_gather_base_mm.default:
            new_shape[dim] = new_shape[dim] // world_size
        elif op_call == npu.npu_mm_reduce_scatter_base.default:
            new_shape[dim] = new_shape[dim] * world_size
        return TensorMeta(shape=torch.Size(new_shape), stride=tensor_meta.stride, dtype=tensor_meta.dtype)

    if op_call == npu.npu_all_gather_base_mm.default:
        world_size = args[3]
        for spec in output_sharding.output_spec: # output, gather_out
            spec.tensor_meta = get_output_meta(spec.tensor_meta, 0, world_size)
    elif op_call == npu.npu_mm_reduce_scatter_base.default:
        world_size = args[3]
        spec = output_sharding.output_spec
        spec.tensor_meta = get_output_meta(spec.tensor_meta, 0, world_size)

    mesh = op_info.compute_mesh
    participating = mesh.get_coordinate() is not None
    if participating:
        # computation that happens in the current rank of the mesh, normal case
        local_args = get_redistributed_local_args(op_info, output_sharding)
        local_kwargs = op_info.local_kwargs
        if op_call == npu.npu_all_gather_base_mm.default:
            local_kwargs = get_redistributed_local_kwargs(
                _infer_npu_all_gather_base_mm_kwargs, op_info, output_sharding
            )
        elif op_call == npu.npu_mm_reduce_scatter_base.default:
            local_kwargs = get_redistributed_local_kwargs(
                _infer_npu_mm_reduce_scatter_base_kwargs, op_info, output_sharding
            )

        local_results = op_call(*local_args, **local_kwargs)
    else:
        # For a non-participating device (happens on rank that does not belong to the device mesh),
        # return empty tensor(s) with correct dtype.
        local_results = get_empty_local_results(op_info, output_sharding)

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


@register_op_strategy(
    [npu.npu_apply_adam_w.default, npu.npu_apply_adam_w.out]
)
def npu_apply_adam_w_strategy(op_schema: OpSchema) -> OpStrategy:
    # npu_apply_adam_w(
    #   Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar weight_decay, Scalar beta1, Scalar beta2,
    #   Scalar epsilon, Tensor grad, Tensor? max_grad_norm, bool? amsgrad, bool? maximize
    # ) -> (Tensor, Tensor, Tensor)
    grad_arg_index = 7
    max_gard_norm_arg_index = 8
    grad_strategy: OpStrategy = op_schema.args_schema[grad_arg_index]
    if "out" in op_schema.kwargs_schema.keys():
        grad_spec: DTensorSpec = op_schema.kwargs_schema["out"].childs[0].strategies[0].output_spec
    else:
        grad_spec: DTensorSpec = grad_strategy.strategies[0].output_spec
    input_target_specs = []
    for i, spec in enumerate(op_schema.args_schema):
        if i == grad_arg_index:
            input_target_specs.append(grad_spec)
        # max_grad_norm follows grad's placements
        elif i == max_gard_norm_arg_index and spec is not None:
            input_target_specs.append(
                DTensorSpec(
                    mesh=grad_spec.mesh,
                    placements=grad_spec.placements,
                    tensor_meta=spec.tensor_meta,
                )
            )
        # only need to provide specs for tensor args (schema is converted to OpStrategy by ShardingPropagator)
        # propagate_op_sharding_non_cached will process non-tensor args
        elif isinstance(spec, OpStrategy):
            input_target_specs.append(spec.strategies[0].output_spec)

    output_spec = []
    for k, values in op_schema.kwargs_schema.items():
        if k == 'out':
            for v in values.childs:
                output_spec.append(v.strategies[0].output_spec)
    output_strategy = OpStrategy([
        PlacementStrategy(output_specs=tuple(output_spec), input_specs=input_target_specs)
    ])

    return output_strategy


def _npu_apply_adam_w_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    # since upwrap_to_op_info does not handle List[DTensor] in kwargs, we need to post-process it here
    _handle_tensor_list_in_kwargs(kwargs, op_info)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding

    mesh = op_info.compute_mesh
    participating = mesh.get_coordinate() is not None
    if participating:
        # computation that happens in the current rank of the mesh, normal case
        if output_sharding.needs_redistribute:
            DTensor._op_dispatcher.redistribute_local_args(
                op_info,
                output_sharding.redistribute_schema
            )
        local_args = (
                pytree.tree_unflatten(
                    cast(list[object], op_info.local_args), op_info.args_tree_spec
                )
                if op_info.args_tree_spec
                else op_info.local_args
            )

        local_results = torch_npu.npu_apply_adam_w(*local_args, **op_info.local_kwargs)

    if _is_out_variant_op(op_call):
        output_specs = (
            (output_sharding.output_spec,)
            if not isinstance(output_sharding.output_spec, tuple)
            else output_sharding.output_spec
        )
        out_dts = []
        spec_idx = 0
        for argument in op_call._schema.arguments:
            if argument.name == 'out':
                for value in kwargs[argument.name]:
                    out_dt = cast(DTensor, value)
                    out_dt._spec = cast(DTensorSpec, output_specs[spec_idx])
                    out_dts.append(out_dt)
                    spec_idx += 1

        return tuple(out_dts) if len(out_dts) > 1 else out_dts[0]
    else:
        return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


customized_ops = {
    npu.npu_grouped_matmul.default: _npu_grouped_matmul_handler,
    npu.npu_grouped_matmul.List: _npu_grouped_matmul_handler,
    npu.npu_apply_adam_w.out: _npu_apply_adam_w_handler,
    npu.npu_all_gather_base_mm.default: npu_comm_mm_fusion_handler,
    npu.npu_mm_reduce_scatter_base.default: npu_comm_mm_fusion_handler
}

old_handlers = DTensor._op_dispatcher._custom_op_handlers
DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}
