import itertools
from typing import Any, Dict, Tuple

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_module, DTensor, Replicate

import torch_npu

npu = torch.ops.npu


def _npu_fusion_attention_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
) -> object:
    def npu_attention_input_fn(
            mesh: DeviceMesh, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                arg = DTensor.from_local(arg, mesh, [Replicate()], run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0: len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args):]))

        return new_args, new_kwargs

    runtime_schema_info = (
        DTensor._op_dispatcher.sharding_propagator.op_to_schema_info.get(op_call, None)
    )

    if runtime_schema_info is not None and runtime_schema_info.needs_pytree:
        try:
            from torch.utils import _cxx_pytree as pytree
        except ImportError:
            from torch.utils import _pytree as pytree  # type: ignore[no-redef]
        from typing import Sequence

        tree_args, args_spec = pytree.tree_flatten(args)
        args_list: Sequence[object] = tree_args
    else:
        args_list, args_spec = args, None

    args, kwargs = npu_attention_input_fn(args_list[0].device_mesh, *args, **kwargs)

    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding

    if op_call == npu.npu_fusion_attention.default:
        local_results = torch_npu.npu_fusion_attention(
            *op_info.local_args, **op_info.local_kwargs
        )
    elif op_call == npu.npu_fusion_attention_grad.default:
        local_results = torch_npu.npu_fusion_attention_grad(
            *op_info.local_args, **op_info.local_kwargs
        )
    else:
        raise NotImplementedError(
            "_npu_fusion_attention_handler only supports npu_fusion_attention and npu_fusion_attention_grad now."
        )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


customized_ops = {
    npu.npu_fusion_attention.default: _npu_fusion_attention_handler,
    npu.npu_fusion_attention_grad.default: _npu_fusion_attention_handler,
}

old_handlers = DTensor._op_dispatcher._custom_op_handlers
DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}
