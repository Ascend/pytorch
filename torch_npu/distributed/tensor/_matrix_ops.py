from typing import Dict, List, Tuple, cast

import torch
from torch.distributed._tensor.experimental import register_sharding
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor._ops.utils import register_op_strategy
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpStrategy,
    OpSpec,
    OutputSharding
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
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], DTensor):
            new_schema = []
            new_local_tensors = []
            for dtensor in value:
                new_schema.append(dtensor._spec)
                new_local_tensors.append(dtensor._local_tensor)
            op_info.schema.kwargs_schema[key] = new_schema
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
        grad_spec: DTensorSpec = op_schema.kwargs_schema["out"].children[0].strategies[0].output_spec
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
            for v in values.children:
                output_spec.append(v.strategies[0].output_spec)
    output_strategy = OpStrategy([
        OpSpec(output_specs=tuple(output_spec), input_specs=input_target_specs)
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
                output_sharding.redistribute_schema,
                output_sharding.use_val_from_redistribute_schema,
            )
        local_args = (
                pytree.tree_unflatten(
                    cast(list[object], op_info.local_args), op_info.args_tree_spec
                )
                if op_info.args_tree_spec
                else op_info.local_args
            )

        local_results = torch_npu.npu_apply_adam_w(*local_args, **op_info.local_kwargs)

    if op_info.schema.is_out_variant_op():
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
    npu.npu_apply_adam_w.out: _npu_apply_adam_w_handler,
    npu.npu_all_gather_base_mm.default: npu_comm_mm_fusion_handler,
    npu.npu_mm_reduce_scatter_base.default: npu_comm_mm_fusion_handler
}

old_handlers = DTensor._op_dispatcher._custom_op_handlers
DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}
