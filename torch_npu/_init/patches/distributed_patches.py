import torch
import torch.distributed.launcher.api

import torch_npu
from torch_npu._init.patches.patch_manager import PatchManager


# 1. Replace PyTorch internal distributed implementations with NPU/HCCL implementations.
_INTERNAL_REPLACEMENTS = [
    (
        "_C._distributed_c10d._verify_params_across_processes",
        "distributed._verify_params_across_processes",
    ),
    (
        "_C._distributed_c10d.ProcessGroup._get_sequence_number_for_group",
        "distributed.distributed_c10d._hccl_get_sequence_number_for_group",
    ),
    (
        "distributed.distributed_c10d._add_ephemeral_timeout_for_all_pgs",
        "distributed.distributed_c10d._hccl_add_ephemeral_timeout_for_all_pgs",
    ),
]


# 2. Expose torch_npu distributed implementations through torch.distributed APIs.
_PUBLIC_API_ALIASES = [
    (
        "distributed.batch_isend_irecv",
        "distributed.distributed_c10d._batch_isend_irecv",
    ),
    (
        "distributed.distributed_c10d.batch_isend_irecv",
        "distributed.distributed_c10d._batch_isend_irecv",
    ),
    (
        "distributed.gather",
        "distributed.distributed_c10d._gather",
    ),
    (
        "distributed.distributed_c10d.gather",
        "distributed.distributed_c10d._gather",
    ),
    (
        "distributed.gather_object",
        "distributed.distributed_c10d._gather_object",
    ),
    (
        "distributed.distributed_c10d.gather_object",
        "distributed.distributed_c10d._gather_object",
    ),
    (
        "distributed.is_hccl_available",
        "distributed.is_hccl_available",
    ),
    (
        "distributed.reinit_process_group",
        "distributed.reinit_process_group",
    ),
]


def _resolve_attr(root, attr_path: str):
    obj = root
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _assign_attr(target_root, target_path: str, source_root, source_path: str):
    parts = target_path.split(".")
    owner = (
        _resolve_attr(target_root, ".".join(parts[:-1]))
        if len(parts) > 1
        else target_root
    )
    setattr(owner, parts[-1], _resolve_attr(source_root, source_path))


def _apply_internal_replacements(torch, torch_npu):
    """
    Replace PyTorch internal distributed implementations with NPU/HCCL versions.

    Example:
        torch._C._distributed_c10d._verify_params_across_processes
            -> torch_npu.distributed._verify_params_across_processes
    """
    for target_path, source_path in _INTERNAL_REPLACEMENTS:
        _assign_attr(torch, target_path, torch_npu, source_path)


def _apply_public_api_aliases(torch, torch_npu):
    """
    Patch torch.distributed public APIs with torch_npu implementations.

    Example:
        torch.distributed.gather(...) -> torch_npu.distributed.distributed_c10d._gather(...)
    """
    for target_path, source_path in _PUBLIC_API_ALIASES:
        _assign_attr(torch, target_path, torch_npu, source_path)


def _apply_wrapped_functions(torch, torch_npu):
    """
    Wrap PyTorch distributed helpers with torch_npu NPU/HCCL logic.

    Example:
        torch.distributed.distributed_c10d.rendezvous(...)
            -> torch_npu.distributed.distributed_c10d._trigger_rendezvous_decorator(...)

        torch.distributed.launcher.api._get_addr_and_port(...)
            -> torch_npu.distributed.distributed_c10d._trigger__get_addr_and_port_decorator(...)
    """
    torch.distributed.distributed_c10d.rendezvous = (
        torch_npu.distributed.distributed_c10d._trigger_rendezvous_decorator(
            torch.distributed.distributed_c10d.rendezvous
        )
    )

    torch.distributed.launcher.api._get_addr_and_port = (
        torch_npu.distributed.distributed_c10d._trigger__get_addr_and_port_decorator(
            torch.distributed.launcher.api._get_addr_and_port
        )
    )


def _apply_sharded_grad_scaler_patch(torch):
    """
    Replace PyTorch FSDP ShardedGradScaler with torch_npu implementation.

    Example:
        torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler
            -> torch_npu.npu.amp.sharded_grad_scaler._ShardedGradScaler
    """
    from torch.distributed.fsdp import sharded_grad_scaler

    from torch_npu.npu.amp.sharded_grad_scaler import _ShardedGradScaler

    sharded_grad_scaler.ShardedGradScaler = _ShardedGradScaler


@PatchManager.register_patch("distributed")
def apply_distributed_methods_patch():
    """
    Patch PyTorch distributed APIs with torch_npu NPU/HCCL implementations.

    Categories:
    1. Internal replacements
    2. Public API aliases
    3. Wrapped functions
    """
    _apply_internal_replacements(torch, torch_npu)
    _apply_public_api_aliases(torch, torch_npu)
    _apply_sharded_grad_scaler_patch(torch)
    _apply_wrapped_functions(torch, torch_npu)
