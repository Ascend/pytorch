from .module import apply_module_patch
from .tensor_methods import add_tensor_methods
from .storage import add_storage_methods
from .combine_tensors import npu_combine_tensors, get_part_combined_tensor, is_combined_tensor_valid
from ._device import apply_device_patch
from .serialization import add_serialization_methods
from .npu_intercept import cann_package_check, add_intercept_methods
from .dtensor import register_ops_under_dtensor_rules
from .collect_env import add_collect_env_methods
from ._dynamo import add_dynamo_methods
from ._dynamo_device import _dynamo_register_interface_for_device
from ._optim import add_optim_method

__all__ = []
