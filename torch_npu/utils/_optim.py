import torch
import torch.optim.optimizer as opt
import torch_npu


_device_name = None


def patch_supported_devices():
    global _device_name
    _device_name = (_device_name if _device_name is not None 
                    else torch_npu.npu.get_device_name(torch_npu.npu.current_device()))

    if _device_name > "Ascend910B" and _device_name < "Ascend910PremiumA":
        return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]
    
    return ["cuda", "xpu"] 


def add_optim_method():
    opt._get_foreach_kernels_supported_devices = patch_supported_devices
