import inspect
import math
import re
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
        and bq == 1
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
    if (
        not w_sparse
        or batch_size != 1
        or num_core <= 0
        or sparse_kv_multiple <= 0
    ):
        return False

    total_base = batch_size * num_kv_heads * num_kv_blocks
    total_weight = num_kv_heads * sum(w_sparse)
    if total_base == 0 or total_weight == 0:
        return False

    mean_weight = total_weight / total_base
    full_rounds, tail_cores = divmod(total_base, num_core)
    has_significant_tail = (
        tail_cores > 0 and full_rounds <= 2 and tail_cores / num_core < 0.5
    )
    has_weight_imbalance = (
        tail_cores == 0 and max(w_sparse) / mean_weight > 1.5
    )
    return has_significant_tail or has_weight_imbalance


def bin_pack_dkdv_hkv_continuous(work_items, num_core):
    bins = [[] for _ in range(num_core)]
    bin_weights = [0.0] * num_core
    groups = {}
    for item in work_items:
        groups.setdefault(item[0], []).append(item)

    for kv_head in sorted(groups):
        group = sorted(
            groups[kv_head], key=lambda item: item[5], reverse=True
        )
        for item in group:
            lightest = bin_weights.index(min(bin_weights))
            bins[lightest].append(item)
            bin_weights[lightest] += item[5]
    return bins


def build_dkdv_task_list(
    w_sparse,
    batch_size,
    num_kv_heads,
    num_kv_blocks,
    sparse_kv_multiple,
    num_core,
):
    target = max(
        num_kv_heads * sum(w_sparse) / max(num_core, 1),
        1.0,
    )
    target_int = max(int(target), 1)
    weights_per_kv_block = [
        int(w_sparse[kv_block // sparse_kv_multiple])
        for kv_block in range(num_kv_blocks)
    ]
    template_items = []
    template_split_bases = []
    max_sub = 1
    for kv_block, weight in enumerate(weights_per_kv_block):
        if weight == 0:
            continue
        if weight <= target:
            template_items.append(
                (kv_block, 0, 1, 0, float(weight))
            )
            continue

        split_count = max(1, math.ceil(weight / target_int))
        template_split_bases.append((kv_block, split_count))
        max_sub = max(max_sub, split_count)
        split_weight = weight / split_count
        for sub_id in range(split_count):
            template_items.append(
                (kv_block, sub_id, split_count, 1, split_weight)
            )

    weighted_items = []
    split_bases = []
    for batch_idx in range(batch_size):
        for kv_head in range(num_kv_heads):
            weighted_items.extend(
                (kv_head, *item) for item in template_items
            )
            split_bases.extend(
                (kv_head, *item)
                for item in template_split_bases
            )

    bins = bin_pack_dkdv_hkv_continuous(weighted_items, num_core)
    work_items = []
    task_offsets = [0]
    for bin_items in bins:
        work_items.extend(item[:5] for item in bin_items)
        task_offsets.append(len(work_items))
    return work_items, task_offsets, split_bases, max_sub


def get_or_build_dkdv_task_list(
    q_num_blks,
    full_q_num_blks,
    batch_size,
    num_kv_heads,
    num_kv_blocks,
    sparse_kv_multiple,
    num_core,
    device,
):
    cache_key = (
        batch_size,
        num_kv_heads,
        num_kv_blocks,
        sparse_kv_multiple,
        num_core,
        device,
    )
    try:
        q_num_blks_version = q_num_blks._version
        full_q_num_blks_version = full_q_num_blks._version
    except RuntimeError:
        q_num_blks_version = None
        full_q_num_blks_version = None

    cache = getattr(q_num_blks, "_npu_dkdv_tasklist_cache", None)
    if cache is not None:
        entry = cache.get(cache_key)
        if (
            entry is not None
            and entry[0] is full_q_num_blks
            and entry[1] == q_num_blks_version
            and entry[2] == full_q_num_blks_version
        ):
            return entry[3]

    weights = compute_dkdv_sparse_weights(q_num_blks, full_q_num_blks)
    use_tasklist = should_use_dkdv_tasklist(
        weights,
        batch_size,
        num_kv_heads,
        num_kv_blocks,
        sparse_kv_multiple,
        num_core,
    )
    if use_tasklist:
        work_items, task_offsets, split_bases, max_sub = (
            build_dkdv_task_list(
                weights,
                batch_size,
                num_kv_heads,
                num_kv_blocks,
                sparse_kv_multiple,
                num_core,
            )
        )
        if work_items:
            work_items_tensor = torch.tensor(
                work_items, dtype=torch.int32, device=device
            )
        else:
            work_items_tensor = torch.zeros(
                (0, 5), dtype=torch.int32, device=device
            )
        task_offsets_tensor = torch.tensor(
            task_offsets, dtype=torch.int32, device=device
        )
        if split_bases:
            split_bases_tensor = torch.tensor(
                split_bases, dtype=torch.int32, device=device
            )
        else:
            split_bases_tensor = torch.zeros(
                (0, 3), dtype=torch.int32, device=device
            )
        result = (
            True,
            work_items_tensor,
            task_offsets_tensor,
            split_bases_tensor,
            max_sub,
        )
    else:
        result = (False, None, None, None, 1)

    if q_num_blks_version is not None and full_q_num_blks_version is not None:
        if cache is None:
            cache = {}
            setattr(q_num_blks, "_npu_dkdv_tasklist_cache", cache)  # noqa: B010
        cache[cache_key] = (
            full_q_num_blks,
            q_num_blks_version,
            full_q_num_blks_version,
            result,
        )
    return result


def _generated_helper_source(function, generated_name, replacements=()):
    source = textwrap.dedent(inspect.getsource(function))
    definition = f"def {function.__name__}("
    source = source.replace(definition, f"def {generated_name}(", 1)
    for old_name, new_name in replacements:
        source = re.sub(
            rf"(?<![\w]){re.escape(old_name)}\(",
            f"{new_name}(",
            source,
        )
    return source


DKDV_TASKLIST_HELPER_SOURCE = "\n\n".join(
    (
        _generated_helper_source(
            compute_dkdv_sparse_weights, "_compute_dkdv_sparse_weights"
        ),
        _generated_helper_source(
            should_use_dkdv_tasklist, "_should_use_dkdv_tasklist"
        ),
        _generated_helper_source(
            bin_pack_dkdv_hkv_continuous,
            "_bin_pack_dkdv_hkv_continuous",
        ),
        _generated_helper_source(
            build_dkdv_task_list,
            "_build_dkdv_task_list",
            (
                (
                    "bin_pack_dkdv_hkv_continuous",
                    "_bin_pack_dkdv_hkv_continuous",
                ),
            ),
        ),
        _generated_helper_source(
            get_or_build_dkdv_task_list,
            "_get_or_build_dkdv_task_list",
            (
                (
                    "compute_dkdv_sparse_weights",
                    "_compute_dkdv_sparse_weights",
                ),
                ("should_use_dkdv_tasklist", "_should_use_dkdv_tasklist"),
                ("build_dkdv_task_list", "_build_dkdv_task_list"),
            ),
        ),
    )
)
