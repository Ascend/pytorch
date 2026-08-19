"""Flex attention tiling configuration generation."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch._inductor import config as inductor_config

from .. import config as npu_config

log = npu_config.log


class FlexMode(Enum):
    """Operation mode for Flex Attention."""
    FWD = "fwd"
    BWDDQ = "bwd_dq"
    BWDDKDV = "bwd_dkdv"


@dataclass
class FlexAttentionConfig:
    """Configuration for Flex Attention kernel tiling."""
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


class FlexAttentionConfigGenerator:
    """Generate block tiling candidates without shape or dtype dependencies."""

    BLOCK_SIZE_CANDIDATES = [256, 128, 64, 32, 16]

    def __init__(
        self,
        sparse_q_block_size: Optional[int] = None,
        sparse_kv_block_size: Optional[int] = None,
        mode: FlexMode = FlexMode.FWD,
    ):
        self.sparse_q_block_size = (
            int(sparse_q_block_size) if sparse_q_block_size is not None else None
        )
        self.sparse_kv_block_size = (
            int(sparse_kv_block_size) if sparse_kv_block_size is not None else None
        )
        self.mode = mode

        if self.mode == FlexMode.BWDDQ:
            self.valid_block_m = self._get_valid_block_sizes(
                self.sparse_kv_block_size
            )
            self.valid_block_n = self._get_valid_block_sizes(
                self.sparse_q_block_size
            )
        else:
            self.valid_block_m = self._get_valid_block_sizes(
                self.sparse_q_block_size
            )
            self.valid_block_n = self._get_valid_block_sizes(
                self.sparse_kv_block_size
            )

    def _get_valid_block_sizes(
        self, sparse_block_size: Optional[int]
    ) -> list[int]:
        if sparse_block_size is None:
            return self.BLOCK_SIZE_CANDIDATES.copy()
        if sparse_block_size <= 0:
            raise ValueError(
                f"sparse block size must be positive, got {sparse_block_size}"
            )
        return [
            size
            for size in self.BLOCK_SIZE_CANDIDATES
            if size <= sparse_block_size and sparse_block_size % size == 0
        ]

    def generate_configs(self) -> list[dict]:
        configs = self._generate_block_combinations()
        configs.sort(
            key=lambda cfg: (
                cfg.block_m * cfg.block_n,
                cfg.block_m,
                cfg.block_n,
            ),
            reverse=True,
        )
        return [cfg.to_dict() for cfg in configs]

    def _generate_block_combinations(self) -> list[FlexAttentionConfig]:
        configs = []
        for block_m in self.valid_block_m:
            for block_n in self.valid_block_n:
                if not self._is_valid_block_pair(block_m, block_n):
                    continue
                configs.append(
                    FlexAttentionConfig(
                        block_m=block_m,
                        block_n=block_n,
                        num_warps=4,
                        num_stages=1,
                    )
                )
        return configs

    def _is_valid_block_pair(self, block_m: int, block_n: int) -> bool:
        if self.mode == FlexMode.BWDDQ:
            block_m2 = block_n
            block_n2 = block_m
            return (
                self.sparse_q_block_size is not None
                and self.sparse_kv_block_size is not None
                and block_m2 == self.sparse_q_block_size
                and block_n2 == self.sparse_kv_block_size
            )
        return (
            self._is_sparse_block_compatible(
                self.sparse_q_block_size, block_m
            )
            and self._is_sparse_block_compatible(
                self.sparse_kv_block_size, block_n
            )
        )

    @staticmethod
    def _is_sparse_block_compatible(
        sparse_block_size: Optional[int], block_size: int
    ) -> bool:
        return (
            block_size > 0
            and (
                sparse_block_size is None
                or (
                    block_size <= sparse_block_size
                    and sparse_block_size % block_size == 0
                )
            )
        )


def prefer_max_tiling_without_benchmark() -> bool:
    return (
        not getattr(inductor_config, "max_autotune", False)
        and not getattr(inductor_config, "max_autotune_gemm", False)
        and not getattr(npu_config, "aggresive_autotune", False)
    )


def generate_fwd_candidate_configs(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> list[dict]:
    """Generate valid configs shared by forward mask-in and mask-out."""
    return FlexAttentionConfigGenerator(
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        mode=FlexMode.FWD,
    ).generate_configs()


def build_sparse_mask_candidate_configs(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> list[dict[str, int]]:
    """Generate sparse mask materialize kernel tiling candidates."""
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    if sparse_q_block_size <= 0 or sparse_kv_block_size <= 0:
        raise ValueError(
            "sparse block sizes must be positive, got "
            f"{sparse_q_block_size} and {sparse_kv_block_size}"
        )
    mask_block_m_candidates = [
        block
        for block in (128, 64)
        if block <= sparse_q_block_size and sparse_q_block_size % block == 0
    ]
    mask_block_n_candidates = [
        block
        for block in (128, 64)
        if block <= sparse_kv_block_size and sparse_kv_block_size % block == 0
    ]
    return [
        {
            "MASK_BLOCK_M": block_m,
            "MASK_BLOCK_N": block_n,
            "NUM_Q_SUB_BLOCKS": sparse_q_block_size // block_m,
            "NUM_KV_SUB_BLOCKS": sparse_kv_block_size // block_n,
            "num_warps": 4,
            "num_stages": 1,
        }
        for block_m in mask_block_m_candidates
        for block_n in mask_block_n_candidates
    ]


def generate_bwd_candidate_configs(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    mode: FlexMode,
) -> list[dict]:
    """Generate valid DQ or DKDV configs shared by mask-in and mask-out."""
    if mode not in (FlexMode.BWDDQ, FlexMode.BWDDKDV):
        raise ValueError(f"unsupported backward flex attention mode: {mode}")
    configs = FlexAttentionConfigGenerator(
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        mode=mode,
    ).generate_configs()
    return [
        {
            "BLOCK_M1": cfg["BLOCK_M"],
            "BLOCK_N1": cfg["BLOCK_N"],
            "BLOCK_M2": cfg["BLOCK_N"],
            "BLOCK_N2": cfg["BLOCK_M"],
            "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"],
        }
        for cfg in configs
    ]


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
