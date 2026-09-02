"""torch_npu inductor FX pass extensions."""

__all__ = ["register_fav3_partition_pass"]

from .fav3_partition_pass import register_fav3_partition_pass
from .post_grad import patch_pattern_mm_plus_mm
