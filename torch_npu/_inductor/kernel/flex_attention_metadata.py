from __future__ import annotations

import logging
from typing import Any

import torch

from torch_npu._inductor import config as npu_config

log = npu_config.log


def _metadata_auto_infer_enabled() -> bool:
    """Return whether metadata auto inference is enabled by config."""
    flex_attention_config = getattr(npu_config, "flex_attention", None)
    if flex_attention_config is None:
        return True
    return getattr(flex_attention_config, "metadata_auto_infer", True)


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
_BLOCK_SPARSE_SAFETY_OPTION_KEYS = (
    "NPU_ROWS_GUARANTEED_SAFE",
    "NPU_BLOCKS_ARE_CONTIGUOUS",
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
    }
    options["HAS_FULL_BLOCKS"] = has_full_blocks
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


def _precomputed_block_sparse_safety_options(block_mask: Any) -> dict[str, Any]:
    """Read exact safety diagnostics cached on a BlockMask by the NPU patch."""
    if block_mask is None:
        return {}
    options = getattr(block_mask, "_npu_flex_attention_kernel_options", None)
    if not isinstance(options, dict):
        return {}
    return {
        key: options[key]
        for key in _BLOCK_SPARSE_SAFETY_OPTION_KEYS
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


def _apply_precomputed_block_sparse_safety_options(
    kernel_options: dict[str, Any],
    block_mask: Any,
    context: str,
) -> dict[str, Any]:
    """Forward cached contiguity diagnostics while keeping row safety conservative."""
    updated = dict(kernel_options)
    safety_options = _precomputed_block_sparse_safety_options(block_mask)
    if "NPU_BLOCKS_ARE_CONTIGUOUS" in safety_options:
        updated.setdefault(
            "BLOCKS_ARE_CONTIGUOUS",
            bool(safety_options["NPU_BLOCKS_ARE_CONTIGUOUS"]),
        )
    if safety_options and log.isEnabledFor(logging.INFO):
        log.info(
            "[flex_attention][%s] cached_block_sparse_safety_diagnostics=%s "
            "forwarded_ROWS_GUARANTEED_SAFE=%s forwarded_BLOCKS_ARE_CONTIGUOUS=%s",
            context,
            safety_options,
            updated.get("ROWS_GUARANTEED_SAFE", "<unset>"),
            updated.get("BLOCKS_ARE_CONTIGUOUS", "<unset>"),
        )
    return updated


def _apply_disabled_metadata_defaults(kernel_options: dict[str, Any]) -> dict[str, Any]:
    """Use conservative kernel defaults when metadata auto inference is disabled."""
    updated = dict(kernel_options)
    updated.setdefault("ROWS_GUARANTEED_SAFE", False)
    updated.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
    return updated


def _normalize_block_rows(kv_num_blocks: torch.Tensor, kv_indices: torch.Tensor) -> tuple[list[int], list[list[int]]]:
    """Convert block-sparse row counts and indices into CPU Python lists."""
    counts = kv_num_blocks.to("cpu", dtype=torch.int64).reshape(-1)
    indices = kv_indices.to("cpu", dtype=torch.int64).reshape(-1, kv_indices.shape[-1])
    row_counts = counts.tolist()
    row_indices = [indices[i, :count].tolist() for i, count in enumerate(row_counts)]
    return row_counts, row_indices


def _infer_blocks_are_contiguous_from_tensors(kv_num_blocks: Any, kv_indices: Any) -> bool | str:
    counts_tensor = _try_unwrap_tensor(kv_num_blocks)
    indices_tensor = _try_unwrap_tensor(kv_indices)
    if counts_tensor is None or indices_tensor is None:
        return "Unknown"

    _, row_indices = _normalize_block_rows(counts_tensor, indices_tensor)
    for values in row_indices:
        if len(values) <= 1:
            continue
        if any((right - left) != 1 for left, right in zip(values, values[1:])):
            return False
    return True


def _layer2_fast_prefilter(block_mask: Any) -> tuple[bool, str]:
    """
    Perform Layer 2 safety check using to_dense() fast pre-filter.

    This checks if every Sparse Q-block has at least one valid KV at the
    Sparse Block level (32x32 elements per cell). Cost: ~1ms per mask.

    Args:
        block_mask: A PyTorch BlockMask object with to_dense() method

    Returns:
        tuple[bool, str]: (is_safe, detail_message)
            - is_safe: True if all rows have >=1 valid cell, False otherwise
            - detail_message: Human-readable result for logging

    Note:
        L2 has 40% false positive rate (granularity trap)!
        Must be combined with L3 whitelist for production safety.
        See v4.0 design doc: Level1_Level2_完整设计文档.md §2.1
    """
    try:
        import time as _time
        _t_start = _time.time()
        if log.isEnabledFor(logging.INFO):
            log.info(
                "[meta][L2] start block_mask_type=%s",
                type(block_mask).__name__,
            )

        dense = block_mask.to_dense()
        dense_device = getattr(dense, "device", "<unknown>")
        dense_dtype = getattr(dense, "dtype", "<unknown>")
        dense_shape = tuple(dense.shape) if hasattr(dense, "shape") else "<unknown>"
        if log.isEnabledFor(logging.INFO):
            log.info(
                "[meta][L2] dense_ready shape=%s dense_device=%s dtype=%s elapsed=%.2fms",
                dense_shape,
                dense_device,
                dense_dtype,
                (_time.time() - _t_start) * 1000,
            )

        b, h = 0, 0
        dense_2d = dense[b, h]

        row_has_valid = dense_2d.any(dim=-1)
        l2_safe = row_has_valid.all().item()

        _elapsed = (_time.time() - _t_start) * 1000

        if l2_safe:
            valid_rows = row_has_valid.sum().item()
            total_rows = len(row_has_valid)
            detail = f"L2 SAFE ({valid_rows}/{total_rows} rows valid, {_elapsed:.2f}ms)"
            if log.isEnabledFor(logging.INFO):
                log.info(
                    "[meta][L2] done safe=True valid_rows=%s total_rows=%s "
                    "dense_device=%s elapsed=%.2fms",
                    valid_rows,
                    total_rows,
                    dense_device,
                    _elapsed,
                )
            return True, detail
        else:
            unsafe_rows = (~row_has_valid).nonzero(as_tuple=True)[0].tolist()
            detail = f"L2 UNSAFE ({len(unsafe_rows)} empty rows: {unsafe_rows[:10]}, {_elapsed:.2f}ms)"
            if log.isEnabledFor(logging.INFO):
                log.info(
                    "[meta][L2] done safe=False unsafe_rows=%s total_rows=%s "
                    "dense_device=%s elapsed=%.2fms",
                    unsafe_rows[:20],
                    len(row_has_valid),
                    dense_device,
                    _elapsed,
                )
            return False, detail

    except Exception as exc:
        if log.isEnabledFor(logging.INFO):
            log.info(
                "[meta][L2] done safe=False error=%s: %s",
                type(exc).__name__,
                exc,
            )
        return False, f"L2 ERROR: {type(exc).__name__}: {exc}"


def _get_critical_positions(seqlen: int, block_size: int = 32, sparse_block_size: int = 128) -> list[int]:
    """
    Generate critical boundary positions for Element-Level safety sampling.

    Instead of scanning all N positions (O(N^2)), we sample key boundary
    locations where granularity traps are most likely to occur:
    - Sparse Block boundaries (multiples of 128)
    - Mask Block boundaries (multiples of 32)
    - First/last positions of each region
    - Segment boundaries if applicable

    This reduces verification from O(N^2) to O(N) while maintaining high accuracy.
    Based on v3.0 empirical validation (526 positions sufficient for seqlen=8192).

    Args:
        seqlen: Total sequence length
        block_size: Mask Block size (default 32)
        sparse_block_size: Sparse Block size (default 128)

    Returns:
        list[int]: Sorted list of critical query positions to check
    """
    positions = set()

    num_sparse_blocks = (seqlen + sparse_block_size - 1) // sparse_block_size
    num_blocks_per_sparse = sparse_block_size // block_size

    for sb in range(num_sparse_blocks):
        sb_start = sb * sparse_block_size
        sb_end = min(sb_start + sparse_block_size, seqlen)

        positions.add(sb_start)
        positions.add(min(sb_end - 1, seqlen - 1))

        for bi in range(num_blocks_per_sparse):
            b_start = sb_start + bi * block_size
            b_end = min(b_start + block_size, seqlen)
            if b_start < seqlen:
                positions.add(b_start)
            if b_end - 1 < seqlen and b_end - 1 >= 0:
                positions.add(b_end - 1)

        mid = (sb_start + sb_end) // 2
        if mid < seqlen:
            positions.add(mid)

    for i in range(0, min(10, seqlen)):
        positions.add(i)

    for i in range(max(0, seqlen - 10), seqlen):
        positions.add(i)

    return sorted([p for p in positions if p < seqlen])


def has_any_valid_kv(mask_fn: Any, q_pos: int, seqlen: int, b: int = 0, h: int = 0) -> bool:
    """
    Check if a specific query position has at least one valid KV position.

    This is the core Element-Level (1x1) safety check. For a given query,
    it scans all possible KVs to find at least one valid (q,k) pair according
    to the mask_mod function.

    Args:
        mask_fn: The mask_mod function from create_block_mask()
        q_pos: Query position to check (element-level index)
        seqlen: Total sequence length
        b: Batch index (default 0)
        h: Head index (default 0)

    Returns:
        bool: True if query has >=1 valid KV, False otherwise

    Note:
        This is the definitive safety check that eliminates granularity traps.
        A return of False means this query would produce NaN in attention computation.
    """
    try:
        device = torch.device("cpu")
        b_idx = torch.tensor(b, device=device)
        h_idx = torch.tensor(h, device=device)
        q_idx = torch.tensor(q_pos, device=device)
        for kv_start in range(0, seqlen, 8192):
            kv_idx = torch.arange(kv_start, min(kv_start + 8192, seqlen), device=device)
            result = mask_fn(b_idx, h_idx, q_idx, kv_idx)
            result_tensor = (
                result.to(dtype=torch.bool)
                if isinstance(result, torch.Tensor)
                else torch.as_tensor(result, dtype=torch.bool, device=device)
            )
            if bool(result_tensor.any().item()):
                return True
        return False
    except Exception:
        return False


def _infer_block_mask_seq_lengths(block_mask: Any) -> tuple[int, int] | None:
    seq_lengths = getattr(block_mask, "seq_lengths", None)
    if (
        isinstance(seq_lengths, tuple)
        and len(seq_lengths) == 2
        and all(isinstance(length, int) for length in seq_lengths)
    ):
        return seq_lengths
    return None


def _infer_block_mask_batch_heads(block_mask: Any, counts: torch.Tensor | None) -> tuple[int, int]:
    if counts is not None and counts.ndim >= 3:
        return int(counts.shape[0]), int(counts.shape[1])
    shape = getattr(block_mask, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 4:
        return int(shape[0]), int(shape[1])
    return 1, 1


def _verify_rows_have_valid_kv_tensorized(
    mask_fn: Any,
    *,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    q_chunk_size: int = 256,
    kv_chunk_size: int = 8192,
    max_unsafe: int = 100,
) -> tuple[bool, list[str]]:
    import time as _time
    _t_start = _time.time()
    unsafe_locations: list[str] = []
    q_chunks_per_head = (q_len + q_chunk_size - 1) // q_chunk_size
    kv_chunks_per_q_chunk = (kv_len + kv_chunk_size - 1) // kv_chunk_size
    total_q_chunks = batch_size * num_heads * q_chunks_per_head
    progress_log_every = max(1, total_q_chunks // 16)
    q_chunks_done = 0
    result_device_warned = False

    if log.isEnabledFor(logging.INFO):
        log.info(
            "[meta][L3] scan_start batch=%d heads=%d q_len=%d kv_len=%d "
            "device=%s q_chunk_size=%d kv_chunk_size=%d total_q_chunks=%d "
            "kv_chunks_per_q_chunk=%d max_unsafe=%d",
            batch_size,
            num_heads,
            q_len,
            kv_len,
            device,
            q_chunk_size,
            kv_chunk_size,
            total_q_chunks,
            kv_chunks_per_q_chunk,
            max_unsafe,
        )

    for b_idx_value in range(batch_size):
        b_idx = torch.tensor(b_idx_value, device=device)
        for h_idx_value in range(num_heads):
            h_idx = torch.tensor(h_idx_value, device=device)
            for q_start in range(0, q_len, q_chunk_size):
                q_end = min(q_start + q_chunk_size, q_len)
                q_idx = torch.arange(q_start, q_end, device=device)[:, None]
                row_has_valid = torch.zeros(q_end - q_start, dtype=torch.bool, device=device)

                for kv_start in range(0, kv_len, kv_chunk_size):
                    kv_end = min(kv_start + kv_chunk_size, kv_len)
                    kv_idx = torch.arange(kv_start, kv_end, device=device)[None, :]
                    result = mask_fn(b_idx, h_idx, q_idx, kv_idx)
                    result_tensor = (
                        result.to(dtype=torch.bool)
                        if isinstance(result, torch.Tensor)
                        else torch.as_tensor(result, dtype=torch.bool, device=device)
                    )
                    result_device = getattr(result_tensor, "device", None)
                    if (
                        not result_device_warned
                        and result_device is not None
                        and str(result_device) != str(device)
                    ):
                        log.warning(
                            "[meta][L3] mask_mod result device=%s differs from "
                            "analysis device=%s; metadata verification may not be "
                            "using the expected accelerated path",
                            result_device,
                            device,
                        )
                        result_device_warned = True
                    if result_tensor.ndim == 0:
                        result_tensor = result_tensor.expand(q_end - q_start, kv_end - kv_start)
                    elif tuple(result_tensor.shape) != (q_end - q_start, kv_end - kv_start):
                        result_tensor = torch.broadcast_to(
                            result_tensor,
                            (q_end - q_start, kv_end - kv_start),
                        )
                    row_has_valid |= result_tensor.any(dim=1)
                    if bool(row_has_valid.all().item()):
                        break

                q_chunks_done += 1
                if (
                    log.isEnabledFor(logging.INFO)
                    and (
                        q_chunks_done == 1
                        or q_chunks_done % progress_log_every == 0
                        or q_chunks_done == total_q_chunks
                    )
                ):
                    valid_rows = int(row_has_valid.sum().item())
                    log.info(
                        "[meta][L3] progress q_chunk=%d/%d b=%d h=%d "
                        "q_range=[%d,%d) valid_rows=%d/%d unsafe_seen=%d "
                        "elapsed=%.2fms device=%s",
                        q_chunks_done,
                        total_q_chunks,
                        b_idx_value,
                        h_idx_value,
                        q_start,
                        q_end,
                        valid_rows,
                        q_end - q_start,
                        len(unsafe_locations),
                        (_time.time() - _t_start) * 1000,
                        device,
                    )

                if not bool(row_has_valid.all().item()):
                    bad_rows = (~row_has_valid).nonzero(as_tuple=True)[0].to("cpu").tolist()
                    for bad_row in bad_rows:
                        unsafe_locations.append(
                            f"b={b_idx_value},h={h_idx_value},q={q_start + int(bad_row)}"
                        )
                        if len(unsafe_locations) >= max_unsafe:
                            if log.isEnabledFor(logging.INFO):
                                log.info(
                                    "[meta][L3] stop max_unsafe=%d reached "
                                    "q_chunk=%d/%d elapsed=%.2fms device=%s",
                                    max_unsafe,
                                    q_chunks_done,
                                    total_q_chunks,
                                    (_time.time() - _t_start) * 1000,
                                    device,
                                )
                            return False, unsafe_locations

    if log.isEnabledFor(logging.INFO):
        log.info(
            "[meta][L3] done safe=%s checked_q_chunks=%d/%d unsafe_count=%d "
            "elapsed=%.2fms device=%s",
            len(unsafe_locations) == 0,
            q_chunks_done,
            total_q_chunks,
            len(unsafe_locations),
            (_time.time() - _t_start) * 1000,
            device,
        )
    return len(unsafe_locations) == 0, unsafe_locations


def _verify_element_level_safety(block_mask: Any) -> tuple[bool, str]:
    """
    Perform TRUE Element-Level (1x1) online safety verification.

    This replaces static whitelist lookup with dynamic runtime analysis.
    For each critical query position, it checks whether there exists at least
    one valid KV using the actual mask_mod function.

    Based on v3.0 methodology (verify_all_masks_element_level.py).
    Samples ~526 critical boundary positions instead of full O(N^2) scan.

    Args:
        block_mask: A PyTorch BlockMask object with mask_mod attribute

    Returns:
        tuple[bool, str]: (is_safe, detail_message)
            - is_safe: True if ALL checked queries have valid KVs
            - detail_message: Human-readable result including unsafe count

    Performance:
        - Uses tensorized q_chunk x kv_chunk row reductions.
        - Runs on the BlockMask tensor device, so eager NPU BlockMask creation
          can perform this check on NPU and cache the result.
    """
    import time as _time
    _t_start = _time.time()

    try:
        mask_mod_fn = getattr(block_mask, 'mask_mod', None)
        kv_num_blocks_attr = getattr(block_mask, 'kv_num_blocks', None)

        if mask_mod_fn is None or kv_num_blocks_attr is None:
            elapsed = (_time.time() - _t_start) * 1000
            return False, f"L3 ERROR: missing mask_mod/kv_num_blocks ({elapsed:.2f}ms)"

        counts = _try_unwrap_tensor(kv_num_blocks_attr)
        if counts is None:
            elapsed = (_time.time() - _t_start) * 1000
            return False, f"L3 ERROR: cannot unwrap kv_num_blocks ({elapsed:.2f}ms)"

        seq_lengths = _infer_block_mask_seq_lengths(block_mask)
        if seq_lengths is None:
            elapsed = (_time.time() - _t_start) * 1000
            return False, f"L3 ERROR: missing seq_lengths ({elapsed:.2f}ms)"
        q_len, kv_len = seq_lengths

        if q_len <= 0 or kv_len <= 0 or q_len > 100000 or kv_len > 100000:
            elapsed = (_time.time() - _t_start) * 1000
            return False, f"L3 ERROR: invalid seq_lengths={seq_lengths} ({elapsed:.2f}ms)"

        batch_size, num_heads = _infer_block_mask_batch_heads(block_mask, counts)
        if log.isEnabledFor(logging.INFO):
            log.info(
                "[meta][L3] start block_mask_type=%s batch=%d heads=%d "
                "q_len=%d kv_len=%d device=%s counts_shape=%s counts_dtype=%s",
                type(block_mask).__name__,
                batch_size,
                num_heads,
                q_len,
                kv_len,
                counts.device,
                tuple(counts.shape),
                counts.dtype,
            )
        if counts.device.type != "npu":
            log.warning(
                "[meta][L3] running on CPU device=%s; NPU acceleration is not "
                "active for metadata row-safety verification",
                counts.device,
            )
        is_safe, unsafe_locations = _verify_rows_have_valid_kv_tensorized(
            mask_mod_fn,
            batch_size=batch_size,
            num_heads=num_heads,
            q_len=q_len,
            kv_len=kv_len,
            device=counts.device,
        )

        elapsed = (_time.time() - _t_start) * 1000

        if is_safe:
            total_rows = batch_size * num_heads * q_len
            return True, f"L3 SAFE ({total_rows} rows verified exactly, {elapsed:.2f}ms)"
        else:
            return (
                False,
                f"L3 UNSAFE ({len(unsafe_locations)} rows have no valid KV: "
                f"{unsafe_locations[:20]}..., {elapsed:.2f}ms)"
            )

    except Exception as exc:
        elapsed = (_time.time() - _t_start) * 1000
        return False, f"L3 EXCEPTION: {type(exc).__name__}: {exc} ({elapsed:.2f}ms)"


def infer_eager_block_mask_kernel_options(block_mask: Any) -> dict[str, Any]:
    """Infer and cache eager BlockMask options before Dynamo graph capture."""
    if block_mask is None:
        return {}

    options = _infer_sparse_mask_compact_options(block_mask)
    if not _metadata_auto_infer_enabled():
        options["NPU_ROWS_GUARANTEED_SAFE"] = False
        options["NPU_BLOCKS_ARE_CONTIGUOUS"] = False
        return options

    kv_num_blocks, kv_indices = _extract_block_sparse_tensors(block_mask)
    if kv_num_blocks is None or kv_indices is None:
        return options

    rows_value, _ = _verify_element_level_safety(block_mask)
    options["NPU_ROWS_GUARANTEED_SAFE"] = rows_value

    contiguous_value = _infer_blocks_are_contiguous_from_tensors(kv_num_blocks, kv_indices)
    if contiguous_value != "Unknown":
        options["NPU_BLOCKS_ARE_CONTIGUOUS"] = contiguous_value
    return options


def infer_block_sparse_metadata(kv_num_blocks: Any, kv_indices: Any, block_mask: Any = None) -> dict[str, Any]:
    """
    Infer metadata from block-sparse structures using v4.0 two-layer architecture.

    Architecture (v4.0):
    - Layer 2: Fast Pre-filter via to_dense() (~1ms) - excludes obviously unsafe
    - Layer 3: Element-Level whitelist lookup (O(1)) - final safety decision

    This replaces the previous overly optimistic "Level 0.5" approach that had
    40% false positive rate (causing NaN crashes in production).

    Args:
        kv_num_blocks: Row-level KV block counts tensor
        kv_indices: Row-level KV block index lists
        block_mask: (NEW) Optional BlockMask object for L2/L3 analysis

    Returns:
        dict with keys:
        - rows_guaranteed_safe: bool/"Unknown" (final verdict)
        - blocks_are_contiguous: bool/"Unknown"
        - is_per_head_heterogeneous: bool
        - empty_row_risk_level: "low"/"medium"/"high"
        - l2_result: str (Layer 2 analysis detail for debugging)
        - l3_result: str (Layer 3 whitelist lookup detail for debugging)
        - safety_layer: str ("L2_FAIL"/"L3_WHITELIST"/"L3_CONSERVATIVE"/"LEGACY")
    """
    if not _metadata_auto_infer_enabled():
        return {
            "rows_guaranteed_safe": "Unknown",
            "blocks_are_contiguous": "Unknown",
            "is_per_head_heterogeneous": False,
            "empty_row_risk_level": "medium",
            "l2_result": "DISABLED_BY_CONFIG",
            "l3_result": "DISABLED_BY_CONFIG",
            "safety_layer": "DISABLED",
        }

    if block_mask is None or not callable(getattr(block_mask, "to_dense", None)):
        log.warning(
            "[meta][v4.0] block_mask is missing to_dense(), falling back to legacy Level 0.5 method. "
            "This has 40% false positive rate! Pass block_mask for v4.0 safety."
        )
        counts_tensor = _try_unwrap_tensor(kv_num_blocks)
        indices_tensor = _try_unwrap_tensor(kv_indices)
        if counts_tensor is None or indices_tensor is None:
            return {
                "rows_guaranteed_safe": "Unknown",
                "blocks_are_contiguous": "Unknown",
                "is_per_head_heterogeneous": False,
                "empty_row_risk_level": "medium",
                "l2_result": "NO_BLOCK_MASK",
                "l3_result": "NO_BLOCK_MASK",
                "safety_layer": "LEGACY_FALLBACK",
            }

        row_counts, row_indices = _normalize_block_rows(counts_tensor, indices_tensor)
        rows_guaranteed_safe_legacy = all(count > 0 for count in row_counts)
        blocks_are_contiguous = True
        for values in row_indices:
            if len(values) <= 1:
                continue
            if any((right - left) != 1 for left, right in zip(values, values[1:])):
                blocks_are_contiguous = False
                break

        headwise_counts = counts_tensor.to("cpu", dtype=torch.int64)
        headwise_indices = indices_tensor.to("cpu", dtype=torch.int64)
        is_per_head_heterogeneous = not _heads_share_used_block_entries(
            headwise_counts, headwise_indices
        )

        return {
            "rows_guaranteed_safe": rows_guaranteed_safe_legacy,
            "blocks_are_contiguous": blocks_are_contiguous,
            "is_per_head_heterogeneous": is_per_head_heterogeneous,
            "empty_row_risk_level": "low" if rows_guaranteed_safe_legacy else "high",
            "l2_result": "SKIPPED_NO_BLOCK_MASK",
            "l3_result": "SKIPPED_NO_BLOCK_MASK",
            "safety_layer": "LEGACY_DEPRECATED",
        }

    l2_safe, l2_detail = _layer2_fast_prefilter(block_mask)

    if not l2_safe:
        counts_tensor = _try_unwrap_tensor(kv_num_blocks)
        indices_tensor = _try_unwrap_tensor(kv_indices)
        if counts_tensor is not None and indices_tensor is not None:
            row_counts, row_indices = _normalize_block_rows(counts_tensor, indices_tensor)
            blocks_are_contiguous = True
            for values in row_indices:
                if len(values) <= 1:
                    continue
                if any((right - left) != 1 for left, right in zip(values, values[1:])):
                    blocks_are_contiguous = False
                    break
        else:
            blocks_are_contiguous = "Unknown"

        return {
            "rows_guaranteed_safe": False,
            "blocks_are_contiguous": blocks_are_contiguous,
            "is_per_head_heterogeneous": False,
            "empty_row_risk_level": "high",
            "l2_result": l2_detail,
            "l3_result": "SKIPPED_L2_FAIL",
            "safety_layer": "L2_FAIL_FAST_REJECT",
        }

    l3_safe, l3_detail = _verify_element_level_safety(block_mask)

    if l3_safe:
        counts_tensor = _try_unwrap_tensor(kv_num_blocks)
        indices_tensor = _try_unwrap_tensor(kv_indices)
        if counts_tensor is not None and indices_tensor is not None:
            row_counts, row_indices = _normalize_block_rows(counts_tensor, indices_tensor)
            blocks_are_contiguous = True
            for values in row_indices:
                if len(values) <= 1:
                    continue
                if any((right - left) != 1 for left, right in zip(values, values[1:])):
                    blocks_are_contiguous = False
                    break
        else:
            blocks_are_contiguous = True

        return {
            "rows_guaranteed_safe": True,
            "blocks_are_contiguous": blocks_are_contiguous,
            "is_per_head_heterogeneous": False,
            "empty_row_risk_level": "low",
            "l2_result": l2_detail,
            "l3_result": f"ELEMENT_VERIFIED_SAFE({l3_detail})",
            "safety_layer": "L3_ELEMENT_PASS",
        }
    else:
        counts_tensor = _try_unwrap_tensor(kv_num_blocks)
        indices_tensor = _try_unwrap_tensor(kv_indices)
        if counts_tensor is not None and indices_tensor is not None:
            row_counts, row_indices = _normalize_block_rows(counts_tensor, indices_tensor)
            blocks_are_contiguous = True
            for values in row_indices:
                if len(values) <= 1:
                    continue
                if any((right - left) != 1 for left, right in zip(values, values[1:])):
                    blocks_are_contiguous = False
                    break
        else:
            blocks_are_contiguous = "Unknown"

        return {
            "rows_guaranteed_safe": "Unknown",
            "blocks_are_contiguous": blocks_are_contiguous,
            "is_per_head_heterogeneous": False,
            "empty_row_risk_level": "medium",
            "l2_result": l2_detail,
            "l3_result": f"ELEMENT_UNSAFE({l3_detail})",
            "safety_layer": "L3_ELEMENT_FAIL",
        }


def build_flex_attention_metadata(kv_num_blocks: Any, kv_indices: Any, block_mask: Any = None) -> dict[str, Any]:
    """Build the minimal metadata contract used by flex attention lowering."""
    metadata = infer_block_sparse_metadata(kv_num_blocks, kv_indices, block_mask=block_mask)
    metadata.setdefault("primary_mask_type", "block_sparse")
    metadata.setdefault("is_segment_aware", "Unknown")
    metadata.setdefault("is_pad_safe", "Unknown")
    metadata.setdefault("causal_granularity", "Unknown")
    metadata.setdefault("is_approximate", False)
    metadata.setdefault("block_granularity_mode", "single_block_size")
    metadata.setdefault("has_cross_half_dependency", False)
    return metadata


def _has_block_sparse_kernel_option_override(kernel_options: dict[str, Any]) -> bool:
    """Check whether block-sparse kernel options are already fully specified."""
    return (
        "ROWS_GUARANTEED_SAFE" in kernel_options
        and "BLOCKS_ARE_CONTIGUOUS" in kernel_options
    )


def _extract_block_sparse_tensors(block_mask: Any) -> tuple[Any, Any]:
    """Read block-sparse tensors from a BlockMask-like object."""
    if block_mask is None:
        return None, None
    return getattr(block_mask, "kv_num_blocks", None), getattr(block_mask, "kv_indices", None)


def apply_kernel_options_from_metadata(
    kernel_options: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Apply conservative kernel-option defaults derived from metadata."""
    updated = dict(kernel_options)
    rows_value = metadata.get("rows_guaranteed_safe", "Unknown")
    contiguous_value = metadata.get("blocks_are_contiguous", "Unknown")
    is_per_head_heterogeneous = metadata.get("is_per_head_heterogeneous", False)
    is_approximate = metadata.get("is_approximate", False)
    block_granularity_mode = metadata.get("block_granularity_mode", "single_block_size")
    empty_row_risk_level = metadata.get("empty_row_risk_level", "medium")
    has_cross_half_dependency = metadata.get("has_cross_half_dependency", False)
    causal_granularity = metadata.get("causal_granularity", "Unknown")

    if "ROWS_GUARANTEED_SAFE" not in updated:
        if rows_value is False or empty_row_risk_level == "high":
            updated["ROWS_GUARANTEED_SAFE"] = False
        elif (
            rows_value is True
            and not is_per_head_heterogeneous
            and not is_approximate
            and block_granularity_mode != "multi_block_size"
            and empty_row_risk_level == "low"
        ):
            updated["ROWS_GUARANTEED_SAFE"] = True

    if "BLOCKS_ARE_CONTIGUOUS" not in updated:
        if contiguous_value is False:
            updated["BLOCKS_ARE_CONTIGUOUS"] = False
        elif (
            contiguous_value is True
            and not is_per_head_heterogeneous
            and not is_approximate
            and block_granularity_mode != "multi_block_size"
            and not has_cross_half_dependency
            and causal_granularity != "mixed"
        ):
            updated["BLOCKS_ARE_CONTIGUOUS"] = True

    return updated


def apply_kernel_options_from_eager_block_mask(
    kernel_options: dict[str, Any] | None,
    block_mask: Any,
    context: str = "eager",
    *,
    allow_tensor_analysis: bool = True,
) -> dict[str, Any]:
    """Infer kernel options from an eager BlockMask-like object when tensors are still available."""
    updated = {} if kernel_options is None else dict(kernel_options)
    updated = _apply_sparse_mask_compact_options(
        updated,
        block_mask,
        context,
        allow_tensor_analysis=allow_tensor_analysis,
    )
    updated = _apply_precomputed_block_sparse_safety_options(
        updated,
        block_mask,
        context,
    )
    if not _metadata_auto_infer_enabled():
        return _apply_disabled_metadata_defaults(updated)

    if not allow_tensor_analysis:
        return updated
    if _has_block_sparse_kernel_option_override(updated):
        return updated

    kv_num_blocks, kv_indices = _extract_block_sparse_tensors(block_mask)
    if kv_num_blocks is None or kv_indices is None:
        return updated

    return apply_kernel_options_from_block_sparse_mask(
        updated,
        kv_num_blocks,
        kv_indices,
        block_mask=block_mask,
        context=context,
    )


def apply_kernel_options_from_block_sparse_mask(
    kernel_options: dict[str, Any] | None,
    kv_num_blocks: Any,
    kv_indices: Any,
    block_mask: Any = None,
    context: str = "unknown"
) -> dict[str, Any]:
    """Infer block-sparse metadata and conservatively merge it into kernel options."""
    updated = {} if kernel_options is None else dict(kernel_options)
    if block_mask is not None:
        updated = _apply_sparse_mask_compact_options(
            updated,
            block_mask,
            context,
            allow_tensor_analysis=True,
        )
        updated = _apply_precomputed_block_sparse_safety_options(
            updated,
            block_mask,
            context,
        )
    if not _metadata_auto_infer_enabled():
        return _apply_disabled_metadata_defaults(updated)

    if _has_block_sparse_kernel_option_override(updated):
        return updated

    metadata = build_flex_attention_metadata(kv_num_blocks, kv_indices, block_mask=block_mask)

    updated = apply_kernel_options_from_metadata(updated, metadata)

    if log.isEnabledFor(logging.INFO):
        log.info(
            "[flex_attention][%s] rows_guaranteed_safe=%s blocks_are_contiguous=%s "
            "is_per_head_heterogeneous=%s empty_row_risk_level=%s "
            "ROWS_GUARANTEED_SAFE=%s BLOCKS_ARE_CONTIGUOUS=%s "
            "safety_layer=%s l2_result=%s l3_result=%s",
            context,
            metadata.get("rows_guaranteed_safe", "Unknown"),
            metadata.get("blocks_are_contiguous", "Unknown"),
            metadata.get("is_per_head_heterogeneous", False),
            metadata.get("empty_row_risk_level", "medium"),
            updated.get("ROWS_GUARANTEED_SAFE", "<unset>"),
            updated.get("BLOCKS_ARE_CONTIGUOUS", "<unset>"),
            metadata.get("safety_layer", "UNKNOWN"),
            metadata.get("l2_result", "N/A"),
            metadata.get("l3_result", "N/A"),
        )
    return updated
