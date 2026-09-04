"""FlexAttention device validation patch for NPU (pre-2.11 torch).

On torch < 2.11 the community does not yet support NPU autocast for
flex_attention, so this module only patches ``_validate_device`` to accept
NPU tensors.  The patch is applied via PatchManager at ``import torch_npu`` time.
"""

from __future__ import annotations

__all__ = []


def _patch_flex_attention_device():
    """Patch flex_attention's _validate_device to accept NPU tensors.

    Eager-mode flex_attention validates input device against a CUDA whitelist.
    Replace it with an NPU-aware check so NPU tensors pass through without
    rejection, while still rejecting mixed-device / cross-device inputs.
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
