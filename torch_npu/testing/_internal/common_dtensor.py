from torch._utils import _get_device_module
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


class NPUDTensorTestBase(DTensorTestBase):
    @property
    def device_type(self):
        return "npu"

    @property
    def world_size(self):
        device_count = _get_device_module(self.device_type).device_count()
        device_num = 4
        if device_count > 1:
            device_num = min(device_num, device_count)
        return device_num
