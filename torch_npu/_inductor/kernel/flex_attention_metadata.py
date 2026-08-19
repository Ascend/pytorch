from __future__ import annotations

import logging
from typing import Any

import torch

from torch_npu._inductor import config as npu_config

log = npu_config.log


def _try_unwrap_tensor(value: Any) -> torch.Tensor | None:
    """Extract a torch.Tensor from wrapper objects that expose a data attribute."""
    if isinstance(value, torch.Tensor):
        return value

    data_value = getattr(value, "data", None)
    if data_value is value or data_value is None:
        return None

    return _try_unwrap_tensor(data_value)


_SPARSE_MASK_COMPACT_OPTION_KEYS = (
    "SPARSE_MASK_MAX_NORMAL_BLOCKS",
    "SPARSE_MASK_HEAD_SHARED",
    "SPARSE_MASK_HQ",
    "HAS_FULL_BLOCKS",
)


def _to_cpu_int_tensor(value: Any) -> torch.Tensor | None:
    """Best-effort conversion of a tensor-like value to a CPU int64 tensor."""
    tensor = _try_unwrap_tensor(value)
    if tensor is None:
        return None
    try:
        return tensor.detach().to("cpu", dtype=torch.int64)
    except Exception:
        return None


def _heads_share_used_block_entries(
    num_blocks: torch.Tensor | None, indices: torch.Tensor | None
) -> bool:
    """
    Return True when all heads have identical used sparse-block entries.

    Only the valid prefix of each row is compared because entries after
    num_blocks[b, h, q] are undefined padding in BlockMask.
    """
    if num_blocks is None or indices is None:
        return False
    if num_blocks.ndim < 3 or indices.ndim < 4:
        return True

    batch = int(num_blocks.shape[0])
    heads = int(num_blocks.shape[1])
    rows = int(num_blocks.shape[2])
    capacity = int(indices.shape[-1])
    if heads <= 1:
        return True

    for b_idx in range(batch):
        for q_idx in range(rows):
            ref_count = int(num_blocks[b_idx, 0, q_idx].item())
            ref_count = max(0, min(ref_count, capacity))
            ref_indices = indices[b_idx, 0, q_idx, :ref_count]
            for h_idx in range(1, heads):
                cur_count = int(num_blocks[b_idx, h_idx, q_idx].item())
                cur_count = max(0, min(cur_count, capacity))
                if cur_count != ref_count:
                    return False
                cur_indices = indices[b_idx, h_idx, q_idx, :cur_count]
                if not torch.equal(cur_indices, ref_indices):
                    return False
    return True


def _infer_sparse_mask_compact_options(block_mask: Any) -> dict[str, Any]:
    """
    Infer compact sparse-mask materialization options from eager BlockMask metadata.

    The options specialize the temporary mask buffer only. KV traversal still uses
    the original BlockMask tensors, so failures to inspect simply return no options
    and leave the existing uncompressed shape in place.
    """
    if block_mask is None:
        return {}

    kv_num_blocks = _to_cpu_int_tensor(getattr(block_mask, "kv_num_blocks", None))
    kv_indices = _to_cpu_int_tensor(getattr(block_mask, "kv_indices", None))
    if kv_num_blocks is None or kv_indices is None:
        return {}
    if kv_num_blocks.numel() == 0 or kv_indices.ndim < 4:
        return {}

    metadata_heads = int(kv_num_blocks.shape[1]) if kv_num_blocks.ndim >= 2 else 1
    capacity = int(kv_indices.shape[-1])
    max_normal_blocks = int(kv_num_blocks.max().item())
    max_normal_blocks = max(1, min(max_normal_blocks, capacity))

    partial_heads_shared = _heads_share_used_block_entries(kv_num_blocks, kv_indices)

    full_kv_num_blocks = _to_cpu_int_tensor(
        getattr(block_mask, "full_kv_num_blocks", None)
    )
    full_kv_indices = _to_cpu_int_tensor(getattr(block_mask, "full_kv_indices", None))
    has_full_blocks = bool(
        full_kv_num_blocks is not None
        and full_kv_num_blocks.numel() > 0
        and full_kv_num_blocks.max().item() > 0
    )

    if full_kv_num_blocks is None and full_kv_indices is None:
        full_heads_shared = True
    elif full_kv_num_blocks is None or full_kv_indices is None:
        full_heads_shared = False
    else:
        full_heads_shared = _heads_share_used_block_entries(
            full_kv_num_blocks, full_kv_indices
        )

    head_shared = partial_heads_shared and full_heads_shared
    options = {
        "SPARSE_MASK_MAX_NORMAL_BLOCKS": max_normal_blocks,
        "SPARSE_MASK_HEAD_SHARED": bool(head_shared),
        "SPARSE_MASK_HQ": 1 if head_shared else metadata_heads,
        "HAS_FULL_BLOCKS": has_full_blocks,
    }
    return options


def _precomputed_sparse_mask_compact_options(block_mask: Any) -> dict[str, Any]:
    """Read compact options cached on a BlockMask by the NPU patch, if present."""
    if block_mask is None:
        return {}
    options = getattr(block_mask, "_npu_flex_attention_kernel_options", None)
    if not isinstance(options, dict):
        return {}
    return {
        key: options[key]
        for key in _SPARSE_MASK_COMPACT_OPTION_KEYS
        if key in options
    }


def _apply_sparse_mask_compact_options(
    kernel_options: dict[str, Any],
    block_mask: Any,
    context: str,
    *,
    allow_tensor_analysis: bool,
) -> dict[str, Any]:
    """Merge cached or freshly inferred sparse-mask compact options."""
    updated = dict(kernel_options)
    compact_options = _precomputed_sparse_mask_compact_options(block_mask)
    missing_compact_options = any(
        key not in updated for key in _SPARSE_MASK_COMPACT_OPTION_KEYS
    )
    if allow_tensor_analysis and missing_compact_options:
        compact_options = {
            **_infer_sparse_mask_compact_options(block_mask),
            **compact_options,
        }

    for key, value in compact_options.items():
        updated.setdefault(key, value)

    if compact_options and log.isEnabledFor(logging.INFO):
        log.info(
            "[flex_attention][%s] sparse_mask_compact_options=%s final_hq=%s final_max_blocks=%s",
            context,
            compact_options,
            updated.get("SPARSE_MASK_HQ", "<unset>"),
            updated.get("SPARSE_MASK_MAX_NORMAL_BLOCKS", "<unset>"),
        )
    return updated


def infer_eager_block_mask_kernel_options(block_mask: Any) -> dict[str, Any]:
    """Infer compact sparse-mask options before Dynamo graph capture."""
    return _infer_sparse_mask_compact_options(block_mask)


def apply_kernel_options_from_eager_block_mask(
    kernel_options: dict[str, Any] | None,
    block_mask: Any,
    context: str = "eager",
    *,
    allow_tensor_analysis: bool = True,
) -> dict[str, Any]:
    """Merge compact sparse-mask options from an eager BlockMask-like object."""
    updated = {} if kernel_options is None else dict(kernel_options)
    return _apply_sparse_mask_compact_options(
        updated,
        block_mask,
        context,
        allow_tensor_analysis=allow_tensor_analysis,
    )
