__all__ = [
    "npu_combine_tensors",
    "get_part_combined_tensor",
    "is_combined_tensor_valid",
    "FlopsCounter",
    "set_thread_affinity",
    "reset_thread_affinity",
    "save_async",
    "get_cann_version",
]

from torch_npu.npu.utils import get_cann_version

from .affinity import (
    _reset_thread_affinity as reset_thread_affinity,
    _set_thread_affinity as set_thread_affinity,
)
from .asd_detector import register_asd_hook, set_asd_loss_scale
from .combine_tensors import (
    get_part_combined_tensor,
    is_combined_tensor_valid,
    npu_combine_tensors,
)
from .flops_count import _FlopsCounter as FlopsCounter
from .serialization import save_async
