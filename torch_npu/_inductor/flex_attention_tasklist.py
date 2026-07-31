import inspect
import math
import textwrap
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeTemplateArg:
    name: str
    dtype: torch.dtype
    rank: int
    wrapper_name: str


@dataclass(frozen=True)
class FlexAttentionDkdvDispatchSpec:
    launch_programs: int
    batch_size: int
    num_kv_heads: int
    num_kv_blocks: int
    sparse_kv_multiple: int
    sparse_kv_block_size: int
    block_n1: int
    partial_dk_stride: int
    partial_dv_stride: int


def is_dkdv_tasklist_codegen_compatible(
    *,
    cpp_wrapper,
    aot_mode,
    bq,
    bkv,
    sparse_z,
    sparse_hq,
    sparse_kv_block_size,
    block_n1,
    q_num_blocks_dtype,
    full_q_num_blocks_dtype,
    q_num_blocks_contiguous,
    full_q_num_blocks_contiguous,
    accum_dtype,
):
    static_dimensions = (bq, bkv, sparse_z, sparse_hq)
    return (
        not cpp_wrapper
        and not aot_mode
        and all(isinstance(value, int) for value in static_dimensions)
        and bq == bkv
        and sparse_z == bq
        and sparse_hq == 1
        and block_n1 > 0
        and sparse_kv_block_size % block_n1 == 0
        and q_num_blocks_dtype == torch.int32
        and full_q_num_blocks_dtype == torch.int32
        and q_num_blocks_contiguous
        and full_q_num_blocks_contiguous
        and accum_dtype == torch.float32
    )


def compute_dkdv_sparse_weights(q_num_blks, full_q_num_blks):
    weights = (q_num_blks + full_q_num_blks).reshape(-1)
    return [
        int(value)
        for value in weights.detach().to("cpu", dtype=torch.int64).tolist()
    ]


def should_use_dkdv_tasklist(
    w_sparse,
    batch_size,
    num_kv_heads,
    num_kv_blocks,
    sparse_kv_multiple,
    num_core,
):
    if not w_sparse or num_core <= 0 or sparse_kv_multiple <= 0:
        return False

    base_weights = []
    sparse_blocks_per_batch = len(w_sparse) // max(batch_size, 1)
    for batch_idx in range(batch_size):
        sparse_base = batch_idx * sparse_blocks_per_batch
        for _ in range(num_kv_heads):
            for kv_block in range(num_kv_blocks):
                sparse_idx = sparse_base + kv_block // sparse_kv_multiple
                if sparse_idx >= len(w_sparse):
                    return False
                base_weights.append(w_sparse[sparse_idx])

    total_weight = sum(base_weights)
    if not base_weights or total_weight == 0:
        return False

    mean_weight = total_weight / len(base_weights)
    full_rounds, tail_cores = divmod(len(base_weights), num_core)
    has_significant_tail = (
        tail_cores > 0 and full_rounds <= 2 and tail_cores / num_core < 0.5
    )
    has_weight_imbalance = (
        tail_cores == 0 and max(base_weights) / mean_weight > 1.5
    )
    return has_significant_tail or has_weight_imbalance


def build_dkdv_task_list(
    w_sparse,
    batch_size,
    num_kv_heads,
    num_kv_blocks,
    sparse_kv_multiple,
    num_core,
):
    sparse_blocks_per_batch = len(w_sparse) // max(batch_size, 1)
    weighted_bases = []
    for batch_idx in range(batch_size):
        sparse_base = batch_idx * sparse_blocks_per_batch
        for kv_head in range(num_kv_heads):
            for kv_block in range(num_kv_blocks):
                weight = int(
                    w_sparse[sparse_base + kv_block // sparse_kv_multiple]
                )
                weighted_bases.append((batch_idx, kv_head, kv_block, weight))

    target = max(
        sum(base[3] for base in weighted_bases) / max(num_core, 1),
        1.0,
    )
    target_int = max(int(target), 1)
    weighted_items = []
    split_bases = []
    max_sub = 1
    for batch_idx, kv_head, kv_block, weight in weighted_bases:
        if weight == 0:
            continue
        if weight <= target:
            split_bases.append((batch_idx, kv_head, kv_block, 1))
            weighted_items.append(
                (batch_idx, kv_head, kv_block, 0, 1, 1, float(weight))
            )
            continue

        split_count = max(1, math.ceil(weight / target_int))
        split_bases.append((batch_idx, kv_head, kv_block, split_count))
        max_sub = max(max_sub, split_count)
        split_weight = weight / split_count
        for sub_id in range(split_count):
            weighted_items.append(
                (
                    batch_idx,
                    kv_head,
                    kv_block,
                    sub_id,
                    split_count,
                    1,
                    split_weight,
                )
            )

    weighted_items.sort(key=lambda item: item[6], reverse=True)
    bins = [[] for _ in range(num_core)]
    bin_weights = [0.0] * num_core
    for item in weighted_items:
        for bin_id in range(num_core):
            if bin_weights[bin_id] + item[6] <= target:
                bins[bin_id].append(item)
                bin_weights[bin_id] += item[6]
                break
        else:
            lightest = min(range(num_core), key=bin_weights.__getitem__)
            bins[lightest].append(item)
            bin_weights[lightest] += item[6]

    work_items = []
    task_offsets = [0]
    for bin_items in bins:
        work_items.extend(item[:6] for item in bin_items)
        task_offsets.append(len(work_items))
    return work_items, task_offsets, split_bases, max_sub


def _generated_helper_source(function, generated_name):
    source = textwrap.dedent(inspect.getsource(function))
    definition = f"def {function.__name__}("
    return source.replace(definition, f"def {generated_name}(", 1)


DKDV_TASKLIST_HELPER_SOURCE = "\n\n".join(
    (
        _generated_helper_source(
            compute_dkdv_sparse_weights, "_compute_dkdv_sparse_weights"
        ),
        _generated_helper_source(
            should_use_dkdv_tasklist, "_should_use_dkdv_tasklist"
        ),
        _generated_helper_source(build_dkdv_task_list, "_build_dkdv_task_list"),
    )
)
