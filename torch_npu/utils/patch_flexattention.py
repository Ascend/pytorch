"""Eager-mode autocast dispatch for flex_attention / flex_attention_backward HOPs.

Import this module (directly or via ``transfer_to_npu``) to register NPU
AutocastPrivateUse1 kernels so that ``torch.autocast(device_type="npu")``
works with ``torch.nn.attention.flex_attention`` outside of ``torch.compile``.

The registration functions are idempotent — safe to import from both the
eager (transfer_to_npu) and inductor (torch_npu._inductor.kernel.flex_attention)
paths without conflicts.
"""

from __future__ import annotations

__all__ = []

from typing import Any, Callable

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.flex_attention import (
    flex_attention as flex_attention_hop,
    flex_attention_backward as flex_attention_backward_hop,
)
from torch.amp.autocast_mode import _cast as _autocast_cast


def _flex_attention_autocast_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast FlexAttention inputs using the active NPU autocast dtype.

    Registered as the AutocastPrivateUse1 kernel for the flex_attention HOP.
    When torch.autocast(device_type="npu", dtype=torch.bfloat16) is active,
    the dispatcher routes here.  We cast Q/K/V to the autocast dtype, then
    redispatch with the autocast key excluded to avoid infinite recursion.
    """
    device_type = query.device.type
    autocast_dtype = torch.get_autocast_dtype(device_type)

    query = _autocast_cast(query, device_type, autocast_dtype)
    key = _autocast_cast(key, device_type, autocast_dtype)
    value = _autocast_cast(value, device_type, autocast_dtype)

    autocast_keyset = torch._C.DispatchKeySet(DispatchKey.AutocastPrivateUse1)
    with torch._C._ExcludeDispatchKeyGuard(autocast_keyset):
        return flex_attention_hop(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


def _flex_attention_backward_autocast_npu(
    query, key, value, out, logsumexp,
    grad_out, grad_logsumexp,
    fw_graph, joint_graph,
    block_mask, scale, kernel_options,
    score_mod_other_buffers, mask_mod_other_buffers,
):
    """Backward HOP autocast: redispatch without re-casting saved tensors.

    PyTorch convention: backward runs outside the autocast context; the forward
    graph already recorded the necessary casts during AOTAutograd capture.
    We only exclude the autocast key and redispatch as-is so the backward HOP
    reaches AOTAutograd / Inductor lowering.
    """
    autocast_keyset = torch._C.DispatchKeySet(DispatchKey.AutocastPrivateUse1)
    with torch._C._ExcludeDispatchKeyGuard(autocast_keyset):
        return flex_attention_backward_hop(
            query, key, value, out, logsumexp,
            grad_out, grad_logsumexp,
            fw_graph, joint_graph,
            block_mask, scale, kernel_options,
            score_mod_other_buffers, mask_mod_other_buffers,
        )


def _register_npu_flex_attention_autocast():
    """Register NPU AutocastPrivateUse1 kernels for both HOPs (idempotent).

    Safe to call multiple times (e.g. across repeated imports or module reloads).
    """
    if not flex_attention_hop.has_kernel_for_dispatch_key(
        DispatchKey.AutocastPrivateUse1
    ):
        flex_attention_hop.py_impl(DispatchKey.AutocastPrivateUse1)(
            _flex_attention_autocast_npu
        )

    if not flex_attention_backward_hop.has_kernel_for_dispatch_key(
        DispatchKey.AutocastPrivateUse1
    ):
        flex_attention_backward_hop.py_impl(DispatchKey.AutocastPrivateUse1)(
            _flex_attention_backward_autocast_npu
        )


def _patch_flex_attention_device():
    """Patch flex_attention's _validate_device to accept NPU tensors.

    Eager-mode flex_attention validates input device against a CUDA whitelist.
    Replace it with a no-op so NPU tensors pass through without rejection.
    Idempotent — skips if already patched.
    """
    try:
        from torch.nn.attention import flex_attention as fa_mod
    except ImportError:
        return
    if getattr(fa_mod, '_npu_device_patched', False):
        return

    def _npu_valid_device(query, key, value):
        if query.device.type != "npu":
            return
        if query.device != key.device or query.device != value.device:
            raise ValueError(
                f"Expected query, key, and value to have the same device, "
                f"but got query.device: {query.device}, key.device: {key.device}, "
                f"and value.device: {value.device} instead."
            )

    fa_mod._validate_device = _npu_valid_device
    fa_mod._npu_device_patched = True
