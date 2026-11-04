import torch
import torch.optim.optimizer as opt
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices

import torch_npu


def patch_supported_devices():
    device_name = torch_npu.npu.get_device_properties(
        torch_npu.npu.current_device()
    ).name
    npu_backend = torch._C._get_privateuse1_backend_name()
    supported_devices = [
        device
        for device in _get_foreach_kernels_supported_devices()
        if device != npu_backend
    ]

    if (device_name > "Ascend910B" and device_name < "Ascend910PremiumA") or (
        device_name > "Ascend910_9"
    ):
        supported_devices.append(npu_backend)

    return supported_devices


def add_optim_method():
    opt._get_foreach_kernels_supported_devices = patch_supported_devices
