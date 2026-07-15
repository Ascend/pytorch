"""Config optimizer for FASTA autotuning pipeline.

Activated by environment variable FASTA_CONFIG_OPTIMIZER=1.
Called after config generation, before add_mutibuffer_config().

Stage 2: NPU-Aware pruning
  - 2d. circle_num <= MAX_CIRCLE_NUM (default 4)
  - 2c. sub_numel >= MIN_SUB_NUMEL (default 32) — minimum tile size

Stage 3: Diversity filter
  - Deduplicate by (circle_num, sub_block_pattern)
  - Cap at MAX_CONFIGS via uniform sub_numel sampling
"""

import math
from torch_npu._inductor.experimental.dynamic_filter.dynamic_filter_config import fasta_config_optimizer
from torch_npu._inductor.fasta_autotune import log


MAX_CIRCLE_NUM = 4
MIN_SUB_NUMEL = 32
MAX_CONFIGS = 50


def _get_circle_num(cfg):
    """Get circle_num from a FastAConfig, computing it if needed."""
    if hasattr(cfg, "circle_num") and cfg.circle_num > 0:
        return cfg.circle_num
    # Fallback: compute from kwargs
    kwargs = cfg.kwargs
    blocks = {}
    subs = {}
    for k, v in kwargs.items():
        if isinstance(v, (int, float)):
            if "BLOCK" in k and "SUB" not in k:
                blocks[k.replace("BLOCK", "")] = v
            elif "BLOCK_SUB" in k:
                subs[k.replace("BLOCK_SUB", "")] = v
    circle = 1
    for axis in blocks:
        if axis in subs and subs[axis] > 0:
            circle *= math.ceil(blocks[axis] / subs[axis])
    return circle


def _get_sub_numel(cfg):
    """Get product of all BLOCK_SUB values (tile footprint in elements)."""
    sub_numel = 1
    for k, v in cfg.kwargs.items():
        if "BLOCK_SUB" in k and isinstance(v, (int, float)) and v > 0:
            sub_numel *= int(v)
    return sub_numel


def _get_sub_pattern(cfg):
    """Get sorted tuple of (key, value) for all BLOCK_SUB params."""
    return tuple(
        sorted(
            (k, int(v))
            for k, v in cfg.kwargs.items()
            if "BLOCK_SUB" in k and isinstance(v, (int, float))
        )
    )


def _dedup_configs(configs):
    """Deduplicate by (circle_num, sub_block_pattern). Keep first seen."""
    seen = {}
    for cfg in configs:
        key = (_get_circle_num(cfg), _get_sub_pattern(cfg))
        if key not in seen:
            seen[key] = cfg
    return list(seen.values())


def _sample_diverse(configs, max_configs):
    """Sample configs uniformly across sub_numel range for diversity.

    Sorts by sub_numel then picks evenly spaced indices to cover
    small, medium, and large tile sizes.
    """
    if len(configs) <= max_configs:
        return configs
    configs_sorted = sorted(configs, key=_get_sub_numel)
    n = len(configs_sorted)
    # Evenly spaced indices including first and last
    indices = set()
    for i in range(max_configs):
        idx = int(i * (n - 1) / (max_configs - 1)) if max_configs > 1 else 0
        indices.add(idx)
    # Sort indices to maintain sub_numel order
    return [configs_sorted[i] for i in sorted(indices)]


def _is_expert(cfg):
    """Check if config originates from base TileGenerator (F0)."""
    return getattr(cfg, "from_expert", False) is True


def optimize_configs(configs):
    """Apply optimization pipeline to config list.

    Expert configs (from base TileGenerator, from_expert=True) are preserved
    unconditionally — they cover the conservative UB budget range that FASTA
    configs may miss. Only FASTA-generated configs go through the filter,
    dedup, and sampling pipeline.

    Pipeline (FASTA configs only):
      1. Circle number filter: keep configs with circle_num <= MAX_CIRCLE_NUM
      2. Minimum tile size: keep configs with sub_numel >= MIN_SUB_NUMEL
      3. Deduplicate by (circle_num, sub_block_pattern)
      4. Cap at MAX_CONFIGS via uniform sub_numel sampling

    Args:
        configs: List of FastAConfig objects

    Returns:
        Filtered list of FastAConfig objects (expert + filtered FASTA)
    """
    if not fasta_config_optimizer or not configs:
        return configs

    before = len(configs)

    # Separate expert configs (from base TileGenerator) from FASTA configs
    expert_configs = [cfg for cfg in configs if _is_expert(cfg)]
    fasta_configs = [cfg for cfg in configs if not _is_expert(cfg)]
    n_expert = len(expert_configs)

    # Apply filters only to FASTA configs
    filtered = [cfg for cfg in fasta_configs if _get_circle_num(cfg) <= MAX_CIRCLE_NUM]

    if MIN_SUB_NUMEL > 1:
        filtered = [cfg for cfg in filtered if _get_sub_numel(cfg) >= MIN_SUB_NUMEL]

    after_filter = len(filtered)

    # Dedup and sample only FASTA configs
    filtered = _dedup_configs(filtered)
    after_dedup = len(filtered)

    if MAX_CONFIGS > 0:
        filtered = _sample_diverse(filtered, MAX_CONFIGS)

    # Merge: expert configs first, then filtered FASTA configs
    result = expert_configs + filtered

    # Safety: never return empty list
    if not result:
        log.warning(
            "config_optimizer: all configs filtered out, keeping original %d", before
        )
        return configs

    after = len(result)
    log.info(
        "config_optimizer: %d -> %d configs "
        "(expert: %d preserved, fasta: %d -> %d -> %d -> %d, "
        "filter: -%d, dedup: -%d, sample: -%d)",
        before,
        after,
        n_expert,
        len(fasta_configs),
        after_filter,
        after_dedup,
        after - n_expert,
        len(fasta_configs) - after_filter,
        after_filter - after_dedup,
        after_dedup - (after - n_expert),
    )

    return result
