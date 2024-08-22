import torch
import torch.optim.optimizer as opt
import torch_npu
from .utils import should_print_warning


def patch_supported_devices():
    device_name = torch_npu.npu.get_device_name(0)
    if device_name > "Ascend910B" and device_name < "Ascend910P":
        return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]
    if should_print_warning():
        print(f"Warning: NPU does not support argument 'foreach' in this device type, "
              f"we set foreach=False by default. Please do not set any value for this argument.")
    return ["cuda", "xpu"] 


def add_optim_method():
    opt._get_foreach_kernels_supported_devices = patch_supported_devices
