from ._module import _apply_module_patch
from .tensor_methods import _add_tensor_methods
from .storage import _add_storage_methods
from .combine_tensors import npu_combine_tensors, get_part_combined_tensor, is_combined_tensor_valid
from ._device import apply_device_patch
from .serialization import _add_serialization_methods, save_async
from .npu_intercept import _cann_package_check, _add_intercept_methods
from .dtensor import _register_ops_under_dtensor_rules
from .collect_env import _add_collect_env_methods
from ._dynamo import add_dynamo_methods
from ._optim import add_optim_method
from .asd_detector import set_asd_loss_scale, register_asd_hook

__all__ = ["save_async", "npu_combine_tensors", "get_part_combined_tensor", "is_combined_tensor_valid",
           "register_asd_hook", "set_asd_loss_scale"]
