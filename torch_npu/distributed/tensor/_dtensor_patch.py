# Copyright (c) Meta Platforms, Inc. and affiliates

import itertools
from functools import partial
from typing import Callable, Optional

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    _is_inplace_op,
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    TupleStrategy
)
from torch.distributed.tensor._ops import utils as dtensor_utils
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable
)
from torch.distributed.tensor.device_mesh import DeviceMesh

try:
    from torch.utils._cxx_pytree import register_pytree_node, tree_leaves
except ImportError:
    from torch.utils._pytree import register_pytree_node, tree_leaves


# Adapted from PyTorch v2.7.1's DTensor experimental register_sharding.
def register_sharding(op):
    """Register an NPU sharding function without importing DTensor experimental APIs."""
    def custom_strategy(custom_sharding_fn, op_schema):
        def strategy_to_spec(strategy):
            if isinstance(strategy, OpStrategy):
                return strategy.strategies[0].output_spec
            if isinstance(strategy, TupleStrategy):
                return tuple(strategy_to_spec(child) for child in strategy.childs)
            return strategy

        args_schema = tuple(strategy_to_spec(arg) for arg in op_schema.args_schema)
        kwargs_schema = {
            key: strategy_to_spec(value)
            for key, value in op_schema.kwargs_schema.items()
        }
        single_mesh_dim_strategies = [
            output_specs + input_specs
            for output_specs, input_specs in custom_sharding_fn(
                *args_schema, **kwargs_schema
            )
        ]
        return dtensor_utils.expand_to_full_mesh_op_strategy(
            op_schema.get_mesh_from_args(),
            op_schema,
            single_mesh_dim_strategies,
            input_index=len(op_schema.op._schema.returns),
            inplace_op=_is_inplace_op(op_schema.op),
        )

    def wrapper(custom_sharding_fn):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            static_argnum = 100
            static_kwargkey = []
            for index, arg in enumerate(overload._schema.arguments):
                if isinstance(arg.type, torch.IntType) or (
                    isinstance(arg.type, torch.OptionalType)
                    and isinstance(arg.type.getElementType(), torch.IntType)
                ):
                    static_argnum = min(index, static_argnum)
                    if arg.kwarg_only:
                        static_kwargkey.append(arg.name)
            schema_info = RuntimeSchemaInfo(
                static_argnum,
                static_kwargkey or None,
                needs_pytree=True,
            )
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                overload,
                partial(custom_strategy, custom_sharding_fn),
                schema_info,
            )
        return custom_sharding_fn

    return wrapper


def _patched_kwargs_strategy(self) -> tuple[OpStrategy, ...]:
    kwargs_vals = (
        tree_leaves(self.kwargs_schema)
        if self.schema_info is not None and self.schema_info.needs_pytree
        else self.kwargs_schema.values()
    )
    return tuple(item for item in kwargs_vals if isinstance(item, OpStrategy))


def _patched_expand_to_full_mesh_op_strategy(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_mesh_dim_strategies: list[PlacementList],
    *,
    input_index: int = 1,
    inplace_op: bool = False,
) -> OpStrategy:
    # Expand the single_mesh_dim_strategies to full mesh dim strategies.
    all_mesh_dim_strategies = [single_mesh_dim_strategies] * mesh.ndim

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list: list[Optional[DTensorSpec]] = []
        for specs in zip(*strategy_comb):
            if specs[0] is not None:
                spec_list.append(DTensorSpec(mesh, specs))
            else:
                spec_list.append(None)

        input_specs: list[DTensorSpec] = [s for s in spec_list[input_index:] if isinstance(s, DTensorSpec)]

        args_strategy = op_schema.args_strategy
        kwargs_strategy = op_schema.kwargs_strategy
        input_args_strategy = args_strategy + kwargs_strategy

        if len(input_specs) != len(input_args_strategy):
            raise AssertionError(
                f"input_specs({len(input_specs)}) != strategies({len(input_args_strategy)}: "
                f"{len(args_strategy)} args + {len(kwargs_strategy)} kwargs)"
            )
        self_spec = input_args_strategy[0].strategies[0].output_spec

        if inplace_op and self_spec.placements != input_specs[0].placements:
            # if it's inplace op, we would only allow the placement strategy to be added when the
            # input_spec matches the first argument's runtime sharding, otherwise we skip
            continue

        # check inputs shardable
        inputs_shardable = all(
            is_tensor_shardable(inp.shape, s)
            for inp, s in zip(input_args_strategy, input_specs)
        )

        # only add to the all_strategies list when all inputs are shardable
        if inputs_shardable:
            redistribute_cost = [
                generate_redistribute_costs(input_strategy, input_spec)
                for input_strategy, input_spec in zip(input_args_strategy, input_specs)
            ]
            if input_index > 1:
                output_specs = tuple(spec_list[:input_index])
            else:
                if spec_list[0] is not None:
                    output_specs = spec_list[0]  # type: ignore[assignment]
                else:
                    raise RuntimeError("output spec is None")
            strategy = PlacementStrategy(
                output_specs=output_specs,
                input_specs=input_specs,
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strategy)

    return OpStrategy(all_strategies)


def _patched_register_tuple_strategy():
    try:
        register_pytree_node(
            TupleStrategy,
            lambda node: (node.childs, None),
            lambda childs, _: TupleStrategy(tuple(childs)),
        )
    except ValueError:
        # already registered TupleStrategy, skip
        pass


def _apply_dtensor_patch():
    # adding kwarg inputs handling in register sharding for previous pytorch version
    # See pytorch/pytorch/pull/168249
    if torch.__version__ < "2.10":
        if not hasattr(OpSchema, "kwargs_strategy"):
            OpSchema.kwargs_strategy = property(_patched_kwargs_strategy)
        torch.distributed.tensor._ops.utils.expand_to_full_mesh_op_strategy = _patched_expand_to_full_mesh_op_strategy

    # register TupleStrategy pytree node to support flattening tensor lists for previous pytorch version
    # See pytorch/pytorch/pull/158046
    if torch.__version__ < "2.9":
        _patched_register_tuple_strategy()


_apply_dtensor_patch()
