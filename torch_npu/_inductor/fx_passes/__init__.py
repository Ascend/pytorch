"""torch_npu inductor FX pass extensions."""

__all__ = ["register_fav3_partition_pass"]

from .fav3_partition_pass import register_fav3_partition_pass
