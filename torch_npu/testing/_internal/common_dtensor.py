from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


class NPUDTensorTestBase(DTensorTestBase):
    @property
    def device_type(self):
        return "npu"
