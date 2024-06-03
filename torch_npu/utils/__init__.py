from ._module import _apply_module_patch
from .tensor_methods import _add_tensor_methods
from .storage import _add_storage_methods
from .combine_tensors import npu_combine_tensors, get_part_combined_tensor, is_combined_tensor_valid
from .serialization import _add_serialization_methods
from .npu_intercept import _cann_package_check, _add_intercept_methods
from .dtensor import _register_ops_under_dtensor_rules
from .collect_env import _add_collect_env_methods
from ._dynamo import add_dynamo_methods
from ._dynamo_device import _dynamo_register_interface_for_device
from ._inductor import _inductor_register_device_op_overrides
from ._optim import add_optim_method
from .asd_detector import set_asd_loss_scale, register_asd_hook
from ._step import add_perf_dump_patch

__all__ = []
