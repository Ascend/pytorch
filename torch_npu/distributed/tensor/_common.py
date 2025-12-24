# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, Callable, Dict, Sequence, Tuple

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OutputSharding
)

try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree


def get_redistributed_local_args(
    op_info: OpInfo,
    output_sharding: OutputSharding
) -> Tuple[object, ...]:
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
    local_args = cast(tuple[object, ...], local_args)
    return local_args


def get_redistributed_local_kwargs(
    kwargs_spec_infer_func: Callable[[OpSchema, OutputSharding], Dict[str, DTensorSpec]],
    op_info: OpInfo,
    output_sharding: OutputSharding
) -> None:
    # in ShardingPropagator.propagate, only the redistribution of args is considered:
    # 1. if args do not need redistribute, output_sharding.redistribute_schema is None
    # 2. if args need redistribute, kwargs_schema in output_sharding.redistribute_schema is still from the source input
    #    schema rather than the selected strategy
    # therefore we need infer the correct kwargs spec from output spec before local call
    src_kwargs_spec = op_info.schema.kwargs_schema
    target_kwargs_spec = kwargs_spec_infer_func(op_info.schema, output_sharding)
    new_local_kwargs = {}
    for key, target_spec in target_kwargs_spec.items():
        local_tensor = op_info.local_kwargs[key]
        src_spec = src_kwargs_spec[key]
        if isinstance(target_spec, DTensorSpec) and src_spec.placements != target_spec.placements:
            resharded_local_tensor = redistribute_local_tensor(local_tensor, src_spec, target_spec)
            new_local_kwargs[key] = resharded_local_tensor
        else:
            new_local_kwargs[key] = local_tensor

    return new_local_kwargs


def get_empty_local_results(op_info: OpInfo, output_sharding: OutputSharding) -> object:
    # For a non-participating device (happens on rank that does not belong to the device mesh), we do:
    # 1. if the return type is scalar, set the local result to None.
    # 2. if the return type is Tensor or List[Tensor], return empty tensor(s) with correct dtype.
    spec = output_sharding.output_spec
    ret_list = op_info.schema.op._schema.returns

    # For a scalar return type(i.e. spec is None), the non-participating device has None as its local result
    local_results = None

    if spec is not None:
        def default_tensor(spec: DTensorSpec) -> torch.Tensor:
            if spec.tensor_meta is not None:
                shape = spec.tensor_meta.shape
                dtype = spec.tensor_meta.dtype
                if len(shape) == 0:
                    # scalar tensor
                    return torch.zeros((), dtype=dtype)
                else:
                    # non-scalar tensor
                    return torch.tensor([], dtype=dtype)
            else:
                raise RuntimeError(f"{spec} has no tensor metadata.")

        if isinstance(spec, DTensorSpec):
            # return a Tensor value
            local_results = default_tensor(spec)
        elif isinstance(spec, Sequence):
            # return a List[Tensor] value
            local_results = [default_tensor(s) if s is not None else None for s in spec]
            if not isinstance(local_results, list):
                raise RuntimeError("local_results is not a list")
            if None in local_results:
                ret_type = str(ret_list[0].type)
                raise NotImplementedError(
                    f"return type {ret_type} in DTensor op is not supported"
                )

    return local_results
