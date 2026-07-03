"""
Flex Attention Configuration Generator.

This module provides FlexAttentionConfigGenerator class that dynamically
generates candidate configurations for Flex Attention kernel autotuning.
Similar to TileGenerator, it generates BLOCK_M/BLOCK_N combinations based
on input shapes and hardware constraints.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
from torch._inductor import config as inductor_config

from .. import config as npu_config

log = npu_config.log


class FlexMode(Enum):
    """Operation mode for Flex Attention."""
    FWD = "fwd"
    BWD = "bwd"


@dataclass
class FlexAttentionConfig:
    """Configuration for Flex Attention kernel."""
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    npu_params: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }
        if self.npu_params:
            result.update(self.npu_params)
        return result


class FlexAttentionConfigGenerator:
    """
    Generate candidate configurations for Flex Attention kernel.

    This class dynamically generates BLOCK_M/BLOCK_N combinations based on
    input shapes and hardware constraints, similar to how TileGenerator
    works for general NPU kernels.

    Key features:
    1. Constraint-aware: Only generates configs that satisfy SPARSE constraints
    2. Performance-oriented: Considers wave efficiency and UB constraints
    3. Extensible: Supports num_warps/num_stages/NPU parameter variations

    Example:
        >>> generator = FlexAttentionConfigGenerator(
        ...     query_shape=(1, 8, 1024, 128),
        ...     key_shape=(1, 8, 1024, 128),
        ...     sparse_q_block_size=128,
        ...     sparse_kv_block_size=128,
        ...     dtype=torch.float16,
        ...     num_cube_core=24,
        ...     mode=FlexMode.FWD,
        ... )
        >>> configs = generator.generate_configs()
        >>> print(len(configs))  # 10-20 configs
    """

    BLOCK_SIZE_CANDIDATES = [256, 128, 64, 32, 16]

    MAX_CONFIGS = 30

    def __init__(
        self,
        query_shape: tuple,
        key_shape: tuple,
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
        dtype: torch.dtype,
        num_cube_core: int,
        mode: FlexMode = FlexMode.FWD,
    ):
        """
        Initialize the configuration generator.

        Args:
            query_shape: Shape of query tensor (batch, heads, seq_len_q, head_dim)
            key_shape: Shape of key tensor (batch, heads, seq_len_kv, head_dim)
            sparse_q_block_size: SPARSE_Q_BLOCK_SIZE constraint
            sparse_kv_block_size: SPARSE_KV_BLOCK_SIZE constraint
            dtype: Data type of tensors
            num_cube_core: Number of AICore (cube cores) available
            mode: Forward or backward mode
        """
        self.batch_size = query_shape[0]
        self.num_heads = query_shape[1]
        self.seq_len_q = query_shape[2]
        self.head_dim = query_shape[3]
        self.seq_len_kv = key_shape[2]

        self.sparse_q_block_size = sparse_q_block_size
        self.sparse_kv_block_size = sparse_kv_block_size

        self.dtype = dtype
        self.dtype_bytes = self._get_dtype_bytes(dtype)

        self.num_cube_core = num_cube_core

        self.mode = mode

        self.valid_block_m = self._get_valid_block_sizes(sparse_q_block_size)
        self.valid_block_n = self._get_valid_block_sizes(sparse_kv_block_size)

        self.configs: list[FlexAttentionConfig] = []

    def _get_dtype_bytes(self, dtype: torch.dtype) -> int:
        """Get bytes per element for dtype."""
        dtype_bytes_map = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
        }
        return dtype_bytes_map.get(dtype, 4)

    def _get_valid_block_sizes(self, sparse_block_size: int) -> list[int]:
        """
        Get valid block sizes that divide SPARSE_BLOCK_SIZE.

        For SPARSE_BLOCK_SIZE=128, returns: [128, 64, 32, 16]
        For SPARSE_BLOCK_SIZE=64, returns: [64, 32, 16]
        """
        valid_sizes = []
        for size in self.BLOCK_SIZE_CANDIDATES:
            if sparse_block_size % size == 0:
                valid_sizes.append(size)
        return valid_sizes

    def generate_configs(self) -> list[dict]:
        """
        Generate all candidate configurations.

        Returns:
            List of config dictionaries with BLOCK_M, BLOCK_N, num_warps, num_stages
        """
        self.configs = []

        self._generate_block_combinations()
        self._filter_by_ub_constraint()
        self._add_npu_params()
        self._limit_config_count()

        return [cfg.to_dict() for cfg in self.configs]

    def _generate_block_combinations(self):
        """Generate BLOCK_M x BLOCK_N combinations based on mode."""
        if self.mode == FlexMode.FWD:
            self._generate_fwd_combinations()
        else:
            self._generate_bwd_combinations()

    def _generate_fwd_combinations(self):
        """
        Generate forward pass configurations.

        Strategy:
        1. Start with safe default config (16, 16) for persistent mode compatibility
        2. Add configs with different BLOCK_M/BLOCK_N ratios for autotuning
        3. Consider wave efficiency (programs per AICore)
        """
        # Use (16, 16) as the default config to ensure compilation stability
        # This is especially important for persistent kernel mode which may have
        # stricter constraints on tile sizes
        default_m, default_n = 16, 16

        if default_m in self.valid_block_m and default_n in self.valid_block_n:
            self.configs.append(FlexAttentionConfig(
                block_m=default_m,
                block_n=default_n,
                num_warps=4,
                num_stages=3,
            ))

        seen = {(default_m, default_n)}

        for block_m in self.valid_block_m:
            for block_n in self.valid_block_n:
                if (block_m, block_n) in seen:
                    continue

                programs_m = (self.seq_len_q + block_m - 1) // block_m
                total_programs = programs_m * self.batch_size * self.num_heads
                wave_efficiency = total_programs / self.num_cube_core

                if wave_efficiency >= 0.5 or total_programs <= self.num_cube_core:
                    self.configs.append(FlexAttentionConfig(
                        block_m=block_m,
                        block_n=block_n,
                        num_warps=4,
                        num_stages=3,
                    ))
                    seen.add((block_m, block_n))

    def _generate_bwd_combinations(self):
        """
        Generate backward pass configurations.

        Backward uses BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2.
        Constraint: BLOCK_N1 % BLOCK_M1 == 0
        """
        # Match the backward outputcode target tile first. dkdv and dq lowering
        # may still override their unused BLOCK_* dimensions independently.
        default_m, default_n = 64, 64
        if default_m in self.valid_block_m and default_n in self.valid_block_n:
            self.configs.append(FlexAttentionConfig(
                block_m=default_m,
                block_n=default_n,
                num_warps=4,
                num_stages=1,
            ))

        seen = {(default_m, default_n)}

        for block_m1 in self.valid_block_m:
            for block_n1 in self.valid_block_n:
                if block_n1 % block_m1 != 0:
                    continue

                if (block_m1, block_n1) in seen:
                    continue

                self.configs.append(FlexAttentionConfig(
                    block_m=block_m1,
                    block_n=block_n1,
                    num_warps=4,
                    num_stages=1,
                ))
                seen.add((block_m1, block_n1))

    def _filter_by_ub_constraint(self):
        """
        Filter configs that exceed UB size limit.

        UB usage estimation:
        - Q block: BLOCK_M * head_dim * dtype_bytes
        - K block: BLOCK_N * head_dim * dtype_bytes
        - V block: BLOCK_N * head_dim * dtype_bytes
        - Acc buffer: BLOCK_M * BLOCK_N * 4 (float32)
        """
        filtered = []

        for cfg in self.configs:
            q_ub = cfg.block_m * self.head_dim * self.dtype_bytes
            k_ub = cfg.block_n * self.head_dim * self.dtype_bytes
            v_ub = cfg.block_n * self.head_dim * self.dtype_bytes
            acc_ub = cfg.block_m * cfg.block_n * 4

            total_ub = q_ub + k_ub + v_ub + acc_ub

            if total_ub <= npu_config.ub_size * 0.8:
                filtered.append(cfg)

        self.configs = filtered if filtered else self.configs[:1]

    def _add_npu_params(self):
        """
        Add NPU optimization parameters if enabled.

        Similar to TileGenerator.tune_multibuffer()
        """
        log.info("[flex_attention] NPU optimization enabled: %s", npu_config.flex_attention.enable_npu_optimization)

        if not npu_config.flex_attention.enable_npu_optimization:
            # Even when NPU optimization is disabled, we need to set ENABLE_COMPILE_HINT
            # to avoid NameError in kernel code
            for cfg in self.configs:
                cfg.npu_params = npu_config.apply_flex_attention_npu_params(
                    cfg.npu_params or {},
                    enable=False,
                )
            return

        npu_params = npu_config.apply_flex_attention_npu_params(
            {},
            enable=True,
        )

        log.debug("NPU parameters: %s", npu_params)

        # Keep the original configs as a conservative fallback, then append the
        # NPU-tuned variants so autotuning can still fall back if the enhanced
        # path overflows UB or hits backend bugs.
        new_configs = []
        for cfg in self.configs:
            cfg.npu_params = npu_config.apply_flex_attention_npu_params(
                cfg.npu_params or {},
                enable=False,
            )
            new_configs.append(cfg)
            cfg_with_npu = FlexAttentionConfig(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                npu_params=npu_params.copy(),
            )
            new_configs.append(cfg_with_npu)
        self.configs = new_configs  # Replace instead of append

    def _limit_config_count(self):
        """
        Limit config count to avoid excessive autotuning time.

        Strategy: Select configs with diverse BLOCK_M/BLOCK_N ratios
        """
        if len(self.configs) <= self.MAX_CONFIGS:
            return

        ratio_groups: dict[float, list[FlexAttentionConfig]] = {}
        for cfg in self.configs:
            ratio = cfg.block_m / cfg.block_n
            if ratio not in ratio_groups:
                ratio_groups[ratio] = []
            ratio_groups[ratio].append(cfg)

        selected = []
        per_group = max(1, self.MAX_CONFIGS // len(ratio_groups))
        for group in ratio_groups.values():
            selected.extend(group[:per_group])

        self.configs = selected[:self.MAX_CONFIGS]

    def calculate_wave_efficiency(self, block_m: int, block_n: int) -> tuple[int, float]:
        """
        Calculate wave efficiency for given block sizes.

        Args:
            block_m: BLOCK_M size
            block_n: BLOCK_N size

        Returns:
            Tuple of (waves, efficiency)
        """
        programs_m = (self.seq_len_q + block_m - 1) // block_m
        total_programs = programs_m * self.batch_size * self.num_heads

        waves = (total_programs + self.num_cube_core - 1) // self.num_cube_core

        efficiency = total_programs / (waves * self.num_cube_core) if waves > 0 else 0.0

        return waves, efficiency

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
    generator = FlexAttentionConfigGenerator(
        query_shape=query_shape,
        key_shape=key_shape,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        dtype=dtype,
        num_cube_core=num_cube_core,
        mode=FlexMode.FWD,
    )
    return generator.generate_configs()


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
    generator = FlexAttentionConfigGenerator(
        query_shape=query_shape,
        key_shape=key_shape,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        dtype=dtype,
        num_cube_core=num_cube_core,
        mode=FlexMode.BWD,
    )
    return generator.generate_configs()


def prefer_max_tiling_without_benchmark() -> bool:
    return (
        npu_config.flex_attention.use_config_generator
        and not getattr(inductor_config, "max_autotune", False)
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


def _get_default_fwd_config(dtype: torch.dtype, head_dim: int) -> dict:
    head_dim = int(head_dim)
    config = {
        "num_warps": 4,
        "num_stages": 3,
    }

    if head_dim <= 256:
        config["BLOCK_M"] = 64
        config["BLOCK_N"] = 64
    elif dtype == torch.float32:
        config["BLOCK_M"] = 32
        config["BLOCK_N"] = 16
    else:
        config["BLOCK_M"] = 32
        config["BLOCK_N"] = 32

    return config


def _tune_npu_params(configs: list[dict]) -> list[dict]:
    enable = npu_config.flex_attention.enable_npu_optimization
    if enable:
        npu_params = npu_config.flex_attention.get_npu_compile_hint_params()
        npu_config.log.info(
            f"[flex_attention] NPU compile hint enabled with parameters: {npu_params}"
        )
        log.debug("npu_params: %s", npu_params)

    return [
        npu_config.apply_flex_attention_npu_params(config, enable=enable)
        for config in configs
    ]


def _build_single_fwd_config(
    dtype: torch.dtype,
    head_dim: int,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> list[dict]:
    config = _get_default_fwd_config(dtype, head_dim)
    cfgs = _tune_npu_params([config])
    for cfg in cfgs:
        cfg["BLOCK_M"] = int(sparse_q_block_size)
        cfg["BLOCK_N"] = int(sparse_kv_block_size)
    return cfgs


def generate_fwd_candidate_configs(
    query_shape: tuple,
    key_shape: tuple,
    dtype: torch.dtype,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    num_cube_core: int,
    head_dim: Optional[int] = None,
) -> list[dict]:
    """
    Generate candidate configs for forward flex attention.

    This wrapper owns the generator/fallback policy so the lowering file only
    needs to pass ordinary Python values extracted from IR nodes.
    """
    if npu_config.flex_attention.use_config_generator:
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
        return configs

    if head_dim is None:
        head_dim = int(query_shape[-1])
    return _build_single_fwd_config(
        dtype=dtype,
        head_dim=head_dim,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
    )


def _flex_attention_sparse_mask_block_candidates(sparse_block_size: int) -> list[int]:
    sparse_block_size = int(sparse_block_size)
    if sparse_block_size <= 0:
        raise ValueError(f"sparse block size must be positive, got {sparse_block_size}")

    min_mask_block = min(16, sparse_block_size)
    candidates = []
    mask_block = sparse_block_size
    while mask_block >= min_mask_block:
        candidates.append(mask_block)
        mask_block //= 2

    for fallback_mask_block in (64, 32, 16):
        if fallback_mask_block <= sparse_block_size:
            candidates.append(fallback_mask_block)

    unique_candidates = []
    seen = set()
    for mask_block in sorted(candidates, reverse=True):
        if mask_block in seen:
            continue
        if sparse_block_size % mask_block != 0:
            continue
        seen.add(mask_block)
        unique_candidates.append(mask_block)

    return unique_candidates


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
    if npu_config.flex_attention.use_config_generator:
        configs = _flex_attention_sparse_mask_tiling_configs(
            sparse_q_block_size,
            sparse_kv_block_size,
        )
        if prefer_max_tiling_without_benchmark():
            configs = _sort_sparse_mask_candidate_configs_for_nobench(configs)
        return configs
    return [
        _get_default_sparse_mask_tiling_config(
            sparse_q_block_size,
            sparse_kv_block_size,
        )
    ]


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
) -> dict[str, Union[int, bool]]:
    tile_mix_loop = _sparse_mask_attention_tile_mix_loop(block_n) if enabled else 0
    return {
        "enable_ubuf_saving": bool(
            getattr(npu_config.flex_attention, "enable_ubuf_saving", True)
        ),
        "multibuffer": enabled,
        "unit_flag": True,
        "limit_auto_multi_buffer_only_for_local_buffer": not enabled,
        "set_workspace_multibuffer": 4,
        "tile_mix_vector_loop": tile_mix_loop,
        "tile_mix_cube_loop": tile_mix_loop,
        "ENABLE_COMPILE_HINT": enable_compile_hint if enabled else False,
    }


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
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    max_block = min(128, sparse_q_block_size, sparse_kv_block_size)

    preferred_blocks = []
    block = max_block
    min_block = min(16, max_block)
    while block >= min_block:
        if sparse_q_block_size % block == 0 and sparse_kv_block_size % block == 0:
            preferred_blocks.append(block)
        block //= 2
    if not preferred_blocks:
        preferred_blocks.append(max_block)

    template = configs[0].copy() if configs else {"num_warps": 4, "num_stages": 1}
    ordered_configs = []
    seen_configs = set()
    tiling_keys = ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")

    for block in preferred_blocks:
        cfg = template.copy()
        cfg.update(
            {
                "BLOCK_M1": block,
                "BLOCK_N1": block,
                "BLOCK_M2": block,
                "BLOCK_N2": block,
                "num_warps": cfg.get("num_warps", 4),
                "num_stages": cfg.get("num_stages", 1),
            }
        )
        config_key = tuple(cfg.get(key) for key in tiling_keys)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        ordered_configs.append(cfg)

    for cfg in configs:
        config_key = tuple(cfg.get(key) for key in tiling_keys)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        ordered_configs.append(cfg)
    return ordered_configs


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
