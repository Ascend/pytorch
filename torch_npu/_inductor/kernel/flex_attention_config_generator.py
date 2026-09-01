"""Flex attention tiling configuration generation."""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Optional

from torch._inductor import config as inductor_config

from .. import config as npu_config

log = npu_config.log
NO_SPARSE_BLOCK_SIZE = 1 << 30


class FlexMode(Enum):
    """Operation mode for Flex Attention."""
    FWD = "fwd"
    BWD = "bwd"
    BWDDQ = "bwd_dq"
    BWDDKDV = "bwd_dkdv"


@dataclass
class FlexAttentionConfig:
    """Configuration for Flex Attention kernel tiling."""
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int

    def to_dict(self, mode: FlexMode) -> dict:
        """Convert canonical blocks to the names consumed by a template."""
        common = {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }
        if mode == FlexMode.FWD:
            return {
                "BLOCK_M": self.block_m,
                "BLOCK_N": self.block_n,
                **common,
            }
        if mode == FlexMode.BWD:
            return {
                "BLOCK_M1": self.block_m,
                "BLOCK_N1": self.block_n,
                "BLOCK_M2": self.block_n,
                "BLOCK_N2": self.block_m,
                **common,
            }
        if mode == FlexMode.BWDDQ:
            return {
                "BLOCK_M2": self.block_m,
                "BLOCK_N2": self.block_n,
                **common,
            }
        if mode == FlexMode.BWDDKDV:
            return {
                "BLOCK_M1": self.block_m,
                "BLOCK_N1": self.block_n,
                **common,
            }
        raise ValueError(f"unsupported flex attention mode: {mode}")


class FlexAttentionConfigGenerator:
    """Generate block tiling candidates without shape or dtype dependencies."""

    BLOCK_SIZE_CANDIDATES = [128, 64, 32, 16]

    def __init__(
        self,
        sparse_q_block_size: Optional[int] = None,
        sparse_kv_block_size: Optional[int] = None,
        mode: FlexMode = FlexMode.FWD,
        kernel_options: Optional[dict] = None,
    ):
        self.sparse_q_block_size = self._normalize_sparse_block_size(
            sparse_q_block_size
        )
        self.sparse_kv_block_size = self._normalize_sparse_block_size(
            sparse_kv_block_size
        )
        self.mode = mode
        self.kernel_options = dict(kernel_options or {})

        self.valid_block_m = self._get_valid_block_sizes(
            self.sparse_q_block_size
        )
        self.valid_block_n = self._get_valid_block_sizes(
            self.sparse_kv_block_size
        )
        if self.mode == FlexMode.BWD:
            common_blocks = [
                block
                for block in self.valid_block_m
                if block in self.valid_block_n
            ]
            self.valid_block_m = common_blocks
            self.valid_block_n = common_blocks

        block_m_keys, block_n_keys = self._block_option_keys()
        user_block_m = self._resolve_user_block(block_m_keys)
        user_block_n = self._resolve_user_block(block_n_keys)
        block_m_sparse_sizes = (self.sparse_q_block_size,)
        block_n_sparse_sizes = (self.sparse_kv_block_size,)
        if self.mode == FlexMode.BWD:
            block_m_sparse_sizes += (self.sparse_kv_block_size,)
            block_n_sparse_sizes += (self.sparse_q_block_size,)
        self.valid_block_m = self._apply_user_block(
            self.valid_block_m,
            user_block_m,
            block_m_keys,
            block_m_sparse_sizes,
        )
        self.valid_block_n = self._apply_user_block(
            self.valid_block_n,
            user_block_n,
            block_n_keys,
            block_n_sparse_sizes,
        )

    def _block_option_keys(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.mode == FlexMode.FWD:
            return (("BLOCK_M",), ("BLOCK_N",))
        if self.mode == FlexMode.BWD:
            return (("BLOCK_M1", "BLOCK_N2"), ("BLOCK_N1", "BLOCK_M2"))
        if self.mode == FlexMode.BWDDQ:
            return (("BLOCK_M2",), ("BLOCK_N2",))
        if self.mode == FlexMode.BWDDKDV:
            return (("BLOCK_M1",), ("BLOCK_N1",))
        raise ValueError(f"unsupported flex attention mode: {self.mode}")

    def _resolve_user_block(self, option_keys: tuple[str, ...]) -> Optional[int]:
        user_options = []
        for key in option_keys:
            if key not in self.kernel_options:
                continue
            value = self.kernel_options[key]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{key} must be an integer, got {value!r}")
            block_size = int(value)
            if block_size <= 0 or block_size & (block_size - 1):
                raise ValueError(
                    f"{key} must be a positive power of 2, got {block_size}"
                )
            user_options.append((key, block_size))
        if not user_options:
            return None

        first_key, block_size = user_options[0]
        for key, value in user_options[1:]:
            if value != block_size:
                raise ValueError(
                    "Conflicting kernel options: "
                    f"{first_key}={block_size} and {key}={value}"
                )
        return block_size

    @staticmethod
    def _apply_user_block(
        generated_blocks: list[int],
        user_block: Optional[int],
        option_keys: tuple[str, ...],
        sparse_block_sizes: tuple[Optional[int], ...],
    ) -> list[int]:
        if user_block is None:
            return generated_blocks
        for sparse_block_size in sparse_block_sizes:
            if sparse_block_size is None:
                continue
            if (
                user_block > sparse_block_size
                or sparse_block_size % user_block != 0
            ):
                option_name = "/".join(option_keys)
                raise ValueError(
                    f"{option_name}={user_block} is incompatible with "
                    f"sparse block size {sparse_block_size}"
                )
        return [user_block]

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

    @staticmethod
    def _normalize_sparse_block_size(
        sparse_block_size: Optional[int],
    ) -> Optional[int]:
        if sparse_block_size is None:
            return None
        sparse_block_size = int(sparse_block_size)
        if sparse_block_size == NO_SPARSE_BLOCK_SIZE:
            return None
        return sparse_block_size

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
        return [cfg.to_dict(self.mode) for cfg in configs]

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
                        num_stages=(
                            2 if self.mode == FlexMode.BWDDKDV else 1
                        ),
                    )
                )
        return configs

    def _is_valid_block_pair(self, block_m: int, block_n: int) -> bool:
        if self.mode == FlexMode.BWD:
            return block_n % block_m == 0
        if (
            self.mode == FlexMode.BWDDQ
            and self.sparse_q_block_size is not None
            and self.sparse_kv_block_size is not None
        ):
            return (
                self.sparse_q_block_size % block_m == 0
                and self.sparse_kv_block_size % block_n == 0
            )
        return True


def prefer_max_tiling_without_benchmark() -> bool:
    return (
        not getattr(inductor_config, "max_autotune", False)
        and not getattr(inductor_config, "max_autotune_gemm", False)
        and not getattr(npu_config, "aggresive_autotune", False)
    )


def generate_fwd_candidate_configs(
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    kernel_options: Optional[dict] = None,
) -> list[dict]:
    """Generate valid forward configs."""
    return FlexAttentionConfigGenerator(
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        mode=FlexMode.FWD,
        kernel_options=kernel_options,
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
    if not mask_block_m_candidates or not mask_block_n_candidates:
        raise ValueError(
            "Q and KV block size must be divisible by BLOCK_M and BLOCK_N. We "
            f"got Q_BLOCK_SIZE={sparse_q_block_size} and "
            f"KV_BLOCK_SIZE={sparse_kv_block_size}."
        )
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
    kernel_options: Optional[dict] = None,
) -> list[dict]:
    """Generate final block configs for a fused or split backward template."""
    if mode not in (FlexMode.BWD, FlexMode.BWDDQ, FlexMode.BWDDKDV):
        raise ValueError(f"unsupported backward flex attention mode: {mode}")
    return FlexAttentionConfigGenerator(
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        mode=mode,
        kernel_options=kernel_options,
    ).generate_configs()


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
