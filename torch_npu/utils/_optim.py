import torch
import torch.optim.optimizer as opt
import torch_npu
from torch_npu.utils.collect_env import get_cann_version


_device_name = None
_cann_version = get_cann_version()
_foreach_black_list_for_cann_starts_with = ['8.0.RC1', '8.0.RC2']
_foreach_black_list_for_cann_all = ['not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T5', '8.0.T6', '8.0.T7',
    '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T37', '8.0.T38', '8.0.T39', '8.0.T50', '8.0.T51', '8.0.T52']


def patch_supported_devices():
    global _device_name
    _device_name = (_device_name if _device_name is not None
                    else torch_npu.npu.get_device_name(torch_npu.npu.current_device()))

    global _cann_version
    if _cann_version is None or _cann_version < '8.0' or _cann_version in _foreach_black_list_for_cann_all:
        return ["cuda", "xpu"]

    for ver in _foreach_black_list_for_cann_starts_with:
        if _cann_version.startswith(ver):
            return ["cuda", "xpu"]

    if _device_name > "Ascend910B" and _device_name < "Ascend910PremiumA":
        return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]

    return ["cuda", "xpu"]


def add_optim_method():
    opt._get_foreach_kernels_supported_devices = patch_supported_devices
