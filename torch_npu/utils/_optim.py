import torch
import torch.optim.optimizer as opt
import torch_npu


def patch_supported_devices():
    device_name = torch_npu.npu.get_device_name(torch_npu.npu.current_device())

    if (device_name > "Ascend910B" and device_name < "Ascend910PremiumA") or (device_name > "Ascend910_9"):
        return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]

    return ["cuda", "xpu"]


def add_optim_method():
    opt._get_foreach_kernels_supported_devices = patch_supported_devices
