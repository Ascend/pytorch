"""Independent functional HCCL collectives."""
from __future__ import annotations

import torch
from torch.distributed import _functional_collectives as funcol
from torch.fx.node import has_side_effect


def functional_all_reduce(
    self: torch.Tensor,
    reduceOp: str,
    group: funcol.RANK_TYPES,
    tag: str = "",
    *,
    premul_factor: float | None = None,
) -> torch.Tensor:
    """Reduce into independent output storage without an input snapshot.

    Ascend950 (A5) or later is required, including empty and single-rank calls.
    Configure HCCL_OP_EXPANSION_MODE=CCU_MS before group initialization and
    use a supported topology. HCCL may still fall back internally.
    String SUM/AVG/MIN/MAX support forward within the CCU_MS dtype matrix.
    PREMUL_SUM requires a Python int/float premul_factor and allocates its
    premultiplied send tensor; it supports float16/float32/bfloat16 inputs.
    Other reductions must omit premul_factor. Tensor factors and ReduceOp
    objects are not supported. Complex inputs support SUM/AVG only.
    Group resolution and automatic waiting reuse PyTorch functional all_reduce.
    The HCCL backend is called directly. ProcessGroupWrapper is not supported;
    create the group without TORCH_DISTRIBUTED_DEBUG=DETAIL.
    The caller must protect the input and its aliases until communication completes.
    Clone at the call site when original data must be isolated from HCCL or
    modified before waiting. Layout/dtype conversion may allocate send storage.
    NPU OpHook callbacks can receive caller-owned storage. Keep those callbacks
    read-only, or clone the input before calling if they may modify it.
    Only SUM supports backward, using the standard functional communication path.
    """
    if not isinstance(reduceOp, str):
        raise TypeError("functional_all_reduce: reduceOp must be a string")
    if premul_factor is not None:
        if type(premul_factor) not in (int, float):
            raise TypeError("functional_all_reduce: premul_factor must be a Python int or float")
        premul_factor = float(premul_factor)
    group_name = funcol._resolve_group_name(group, tag)
    output = torch.ops.npu_custom_dist.functional_all_reduce(
        self, reduceOp.lower(), group_name, premul_factor
    )
    return funcol._maybe_wrap_tensor(output)


torch.utils.set_module(functional_all_reduce, "torch_npu.distributed")


@torch.library.register_fake("npu_custom_dist::functional_all_reduce")
def _functional_all_reduce_fake(input, reduce_op, group_name, premul_factor=None):
    # Native output is contiguous, including for noncontiguous input views.
    return torch.empty_like(input, memory_format=torch.contiguous_format)


def _functional_all_reduce_setup_context(ctx, inputs, output):
    _, reduce_op, group_name, _ = inputs
    ctx.group_name = group_name
    ctx.reduce_op = reduce_op


def _functional_all_reduce_backward(ctx, grad_output):
    if ctx.reduce_op != "sum":
        raise RuntimeError(
            "all_reduce backward only supports 'sum' reduction, "
            f"got '{ctx.reduce_op}'"
        )
    grad = torch.ops._c10d_functional.all_reduce(
        grad_output.contiguous(), "sum", ctx.group_name
    )
    grad = torch.ops._c10d_functional.wait_tensor(grad)
    return grad, None, None, None


torch.library.register_autograd(
    "npu_custom_dist::functional_all_reduce",
    _functional_all_reduce_backward,
    setup_context=_functional_all_reduce_setup_context,
)

# Even an unused result participates in collective ordering.
has_side_effect(torch.ops.npu_custom_dist.functional_all_reduce.default)
has_side_effect(torch.ops.npu_custom_dist.functional_all_reduce)
