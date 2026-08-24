"""
Flex Attention Configuration Generator.

The NPU templates use a small fixed tiling set. Shape-dependent tiling
selection is intentionally left to autotuning, matching the community template
interface while keeping NPU-specific compatibility filtering local.
"""

from typing import Optional, Union

import torch
from torch._inductor import config as inductor_config

from .. import config as npu_config

log = npu_config.log

_COMMON_TILING_CANDIDATES = (
    (128, 128),
    (128, 64),
    (64, 128),
    (64, 64),
)


def _fixed_tiling_configs(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    *,
    num_stages: int,
) -> list[dict]:
    """Return the small common tiling set supported by the NPU templates."""
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    return [
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "num_warps": 4,
            "num_stages": num_stages,
            "ENABLE_COMPILE_HINT": False,
        }
        for block_m, block_n in _COMMON_TILING_CANDIDATES
        if (
            block_m <= sparse_q_block_size
            and block_n <= sparse_kv_block_size
            and sparse_q_block_size % block_m == 0
            and sparse_kv_block_size % block_n == 0
        )
    ]


def generate_fwd_configs(
    query_shape: tuple,
    key_shape: tuple,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    dtype: torch.dtype,
    num_cube_core: int,
) -> list[dict]:
    """
    Convenience function to generate forward configs.

    Args:
        query_shape: Shape of query tensor
        key_shape: Shape of key tensor
        sparse_q_block_size: SPARSE_Q_BLOCK_SIZE
        sparse_kv_block_size: SPARSE_KV_BLOCK_SIZE
        dtype: Data type
        num_cube_core: Number of AICore

    Returns:
        List of config dictionaries
    """
    del query_shape, key_shape, dtype, num_cube_core
    return _fixed_tiling_configs(
        sparse_q_block_size,
        sparse_kv_block_size,
        num_stages=1,
    )


def generate_bwd_configs(
    query_shape: tuple,
    key_shape: tuple,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    dtype: torch.dtype,
    num_cube_core: int,
) -> list[dict]:
    """
    Convenience function to generate backward configs.

    Args:
        query_shape: Shape of query tensor
        key_shape: Shape of key tensor
        sparse_q_block_size: SPARSE_Q_BLOCK_SIZE
        sparse_kv_block_size: SPARSE_KV_BLOCK_SIZE
        dtype: Data type
        num_cube_core: Number of AICore

    Returns:
        List of config dictionaries
    """
    del query_shape, key_shape, dtype, num_cube_core
    return [
        config
        for config in _fixed_tiling_configs(
            sparse_q_block_size,
            sparse_kv_block_size,
            num_stages=1,
        )
        if config["BLOCK_N"] >= config["BLOCK_M"]
    ]


def prefer_max_tiling_without_benchmark() -> bool:
    return (
        not getattr(inductor_config, "max_autotune", False)
        and not getattr(inductor_config, "max_autotune_gemm", False)
        and not getattr(npu_config, "aggresive_autotune", False)
    )


def _sort_fwd_candidate_configs_for_nobench(configs: list[dict]) -> list[dict]:
    return sorted(
        configs,
        key=lambda cfg: (
            int(cfg.get("BLOCK_M", 0)) * int(cfg.get("BLOCK_N", 0)),
            int(cfg.get("BLOCK_M", 0)),
            int(cfg.get("BLOCK_N", 0)),
        ),
        reverse=True,
    )


_FWD_MASK_IN_TILING_ORDER = (
    (128, 128),
    (128, 64),
    (64, 128),
    (64, 64),
)


def _build_fwd_mask_in_candidate_configs(
    configs: list[dict],
    *,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> list[dict]:
    template = configs[0].copy() if configs else {
        "num_warps": 4,
        "num_stages": 1,
    }
    ordered_configs = []

    for block_m, block_n in _FWD_MASK_IN_TILING_ORDER:
        if (
            block_m > sparse_q_block_size
            or block_n > sparse_kv_block_size
            or sparse_q_block_size % block_m != 0
            or sparse_kv_block_size % block_n != 0
        ):
            continue
        cfg = template.copy()
        cfg.update(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "num_warps": cfg.get("num_warps", 4),
                "num_stages": 1,
            }
        )
        ordered_configs.append(cfg)

    if ordered_configs:
        return ordered_configs

    return [
        cfg
        for cfg in configs
        if (
            sparse_q_block_size % int(cfg["BLOCK_M"]) == 0
            and sparse_kv_block_size % int(cfg["BLOCK_N"]) == 0
        )
    ]


def _sort_sparse_mask_candidate_configs_for_nobench(
    configs: list[dict[str, int]],
) -> list[dict[str, int]]:
    return sorted(
        configs,
        key=lambda cfg: (
            int(cfg["MASK_BLOCK_M"]) * int(cfg["MASK_BLOCK_N"]),
            int(cfg["MASK_BLOCK_M"]),
            int(cfg["MASK_BLOCK_N"]),
        ),
        reverse=True,
    )


def get_bwd_dq_compile_options() -> dict:
    return npu_config.flex_attention.get_bwd_dq_compile_options()


def get_bwd_dkdv_compile_options() -> dict:
    return npu_config.flex_attention.get_bwd_dkdv_compile_options()


def generate_fwd_candidate_configs(
    query_shape: tuple,
    key_shape: tuple,
    dtype: torch.dtype,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    num_cube_core: int,
    head_dim: Optional[int] = None,
    mask_out: bool = True,
) -> list[dict]:
    """
    Generate candidate configs for forward flex attention.

    This wrapper owns the generator/fallback policy so the lowering file only
    needs to pass ordinary Python values extracted from IR nodes.
    """
    del head_dim
    configs = generate_fwd_configs(
        query_shape=query_shape,
        key_shape=key_shape,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        dtype=dtype,
        num_cube_core=num_cube_core,
    )
    if prefer_max_tiling_without_benchmark():
        configs = _sort_fwd_candidate_configs_for_nobench(configs)
    if not mask_out:
        configs = _build_fwd_mask_in_candidate_configs(
            configs,
            sparse_q_block_size=sparse_q_block_size,
            sparse_kv_block_size=sparse_kv_block_size,
        )
    return configs


def _flex_attention_sparse_mask_block_candidates(sparse_block_size: int) -> list[int]:
    sparse_block_size = int(sparse_block_size)
    if sparse_block_size <= 0:
        raise ValueError(f"sparse block size must be positive, got {sparse_block_size}")

    return [
        block
        for block in (128, 64)
        if block <= sparse_block_size and sparse_block_size % block == 0
    ]


def _flex_attention_sparse_mask_tiling_configs(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> list[dict[str, int]]:
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    if sparse_q_block_size <= 0:
        raise ValueError(
            f"SPARSE_Q_BLOCK_SIZE must be positive, got {sparse_q_block_size}"
        )
    if sparse_kv_block_size <= 0:
        raise ValueError(
            f"SPARSE_KV_BLOCK_SIZE must be positive, got {sparse_kv_block_size}"
        )

    mask_block_m_candidates = _flex_attention_sparse_mask_block_candidates(
        sparse_q_block_size
    )
    mask_block_n_candidates = _flex_attention_sparse_mask_block_candidates(
        sparse_kv_block_size
    )

    configs = []
    seen = set()
    candidate_pairs = (
        (mask_block_m, mask_block_n)
        for mask_block_m in mask_block_m_candidates
        for mask_block_n in mask_block_n_candidates
    )

    for mask_block_m, mask_block_n in candidate_pairs:
        if (mask_block_m, mask_block_n) in seen:
            continue
        seen.add((mask_block_m, mask_block_n))
        configs.append(
            {
                "MASK_BLOCK_M": mask_block_m,
                "MASK_BLOCK_N": mask_block_n,
                "NUM_Q_SUB_BLOCKS": sparse_q_block_size // mask_block_m,
                "NUM_KV_SUB_BLOCKS": sparse_kv_block_size // mask_block_n,
                "num_warps": 4,
                "num_stages": 1,
            }
        )

    return configs


def _get_default_sparse_mask_tiling_config(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> dict[str, int]:
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    return {
        "MASK_BLOCK_M": sparse_q_block_size,
        "MASK_BLOCK_N": sparse_kv_block_size,
        "NUM_Q_SUB_BLOCKS": 1,
        "NUM_KV_SUB_BLOCKS": 1,
        "num_warps": 4,
        "num_stages": 1,
    }


def build_sparse_mask_candidate_configs(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> list[dict[str, int]]:
    """Generate sparse mask materialize kernel tiling candidates."""
    configs = _flex_attention_sparse_mask_tiling_configs(
        sparse_q_block_size,
        sparse_kv_block_size,
    )
    if prefer_max_tiling_without_benchmark():
        configs = _sort_sparse_mask_candidate_configs_for_nobench(configs)
    return configs


def split_attention_block_n_candidates(
    base_block_n: int,
    min_block_n: int = 64,
) -> list[int]:
    base_block_n = int(base_block_n)
    min_block_n = int(min_block_n)
    if base_block_n <= 0:
        raise ValueError(f"base_block_n must be positive, got {base_block_n}")
    if min_block_n <= 0:
        raise ValueError(f"min_block_n must be positive, got {min_block_n}")

    candidates: list[int] = []
    current = base_block_n
    while current >= min_block_n:
        if base_block_n % current == 0:
            candidates.append(current)
        current //= 2

    if not candidates:
        candidates.append(base_block_n)
    return candidates


def _sparse_mask_attention_tile_mix_loop(block_n: int) -> int:
    block_n = int(block_n)
    if block_n >= 512:
        return 4
    if block_n >= 256:
        return 2
    if block_n >= 128:
        return 1
    return 0


def _sparse_mask_attention_cvpipeline_options(
    block_n: int,
    *,
    enabled: bool,
    enable_compile_hint: bool = False,
) -> dict[str, Union[int, bool, str]]:
    tile_mix_loop = _sparse_mask_attention_tile_mix_loop(block_n) if enabled else 0
    return npu_config.flex_attention.get_sparse_mask_cvpipeline_compile_options(
        enabled=enabled,
        tile_mix_loop=tile_mix_loop,
        enable_compile_hint=enable_compile_hint,
    )


def sparse_mask_attention_cvpipeline_config_variants(
    base_options: dict,
    *,
    block_n: int,
    enable_compile_hint: bool = False,
) -> list[dict]:
    variants = []
    for enabled in (True, False):
        variant = base_options.copy()
        variant.update(
            _sparse_mask_attention_cvpipeline_options(
                block_n,
                enabled=enabled,
                enable_compile_hint=enable_compile_hint,
            )
        )
        variants.append(variant)
    return variants


def is_bwd_config_compatible(
    cfg: dict,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> bool:
    block_m1 = cfg["BLOCK_M1"]
    block_n1 = cfg["BLOCK_N1"]
    block_m2 = cfg["BLOCK_M2"]
    block_n2 = cfg["BLOCK_N2"]
    return (
        sparse_q_block_size % block_m1 == 0
        and sparse_kv_block_size % block_n1 == 0
        and sparse_q_block_size % block_m2 == 0
        and sparse_kv_block_size % block_n2 == 0
    )


def _convert_bwd_config_to_fused_mask_out_config(cfg: dict) -> dict:
    converted_cfg = {
        "BLOCK_M1": cfg["BLOCK_M"],
        "BLOCK_N1": cfg["BLOCK_N"],
        "BLOCK_M2": cfg["BLOCK_N"],
        "BLOCK_N2": cfg["BLOCK_M"],
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
    }
    for key, value in cfg.items():
        if key not in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages"):
            converted_cfg[key] = value
    return converted_cfg


def _start_bwd_mask_out_from_128x128_configs(
    configs: list[dict],
    *,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> list[dict]:
    return [
        cfg
        for cfg in configs
        if is_bwd_config_compatible(
            cfg,
            sparse_q_block_size,
            sparse_kv_block_size,
        )
    ]


def generate_bwd_fused_mask_out_candidate_configs(
    query_shape: tuple,
    key_shape: tuple,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    dtype: torch.dtype,
    num_cube_core: int,
) -> list[dict]:
    """
    Generate candidate configs for the fused compact sparse mask-out backward path.

    The fused backward kernel uses the split backward tiling names but runs as a
    single compact sparse mask-out template. Keep 128x128 square tiling first so
    the generated output_code remains aligned with the verified path.
    """
    base_configs = generate_bwd_configs(
        query_shape=query_shape,
        key_shape=key_shape,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        dtype=dtype,
        num_cube_core=num_cube_core,
    )
    configs = [
        _convert_bwd_config_to_fused_mask_out_config(cfg)
        for cfg in base_configs
    ]
    return _start_bwd_mask_out_from_128x128_configs(
        configs,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
    )


def generate_bwd_split_mask_out_candidate_configs(
    query_shape: tuple,
    key_shape: tuple,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    dtype: torch.dtype,
    num_cube_core: int,
) -> list[dict]:
    """Generate candidate configs for split DQ and DKDV backward mask-out kernels."""
    return generate_bwd_fused_mask_out_candidate_configs(
        query_shape=query_shape,
        key_shape=key_shape,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        dtype=dtype,
        num_cube_core=num_cube_core,
    )


def validate_benchmark_config() -> None:
    """
    Validate benchmark configuration before autotuning.

    This function checks that required configurations are enabled for
    NPU optimized benchmark.

    Note: This function now only warns instead of raising errors to avoid
    blocking execution. The actual benchmark will use fallback methods if
    configurations are not optimal.
    """
    aggresive_autotune = getattr(npu_config, 'aggresive_autotune', False)
    max_autotune = getattr(inductor_config, 'max_autotune', False)

    if not aggresive_autotune:
        log.warning(
            "aggresive_autotune is False. NPU optimized benchmark is disabled. "
            "For optimal performance, set INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE=1 environment variable. "
            "Continuing with fallback benchmark method."
        )

    if not max_autotune:
        log.warning(
            "max_autotune is False, only default config will be used. "
            "Set TORCHINDUCTOR_MAX_AUTOTUNE=1 for multi-config autotuning."
        )
