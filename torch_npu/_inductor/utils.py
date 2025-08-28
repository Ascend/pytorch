from typing import Optional

import torch
from torch._inductor import utils, graph, scheduler

import torch_npu

NPU_TYPES = ["npu"]


# Not good implementation, but no other way
def get_current_raw_stream(device):
    return torch.npu.current_stream(device).npu_stream


def is_npu(device: Optional[str]):
    assert isinstance(device, str) or device is None, device
    return device in NPU_TYPES


def patch_device_need_guard():
    def device_need_guard_npu(device: str):
        assert isinstance(device, str)
        return utils.is_gpu(device) or is_npu(device)

    utils.device_need_guard = device_need_guard_npu
    scheduler.device_need_guard = device_need_guard_npu


def patch_is_same_tensor():
    from torch._subclasses.fake_tensor import FakeTensor

    def is_same_tensor(data: torch.Tensor, value: torch.Tensor):
        if isinstance(data, FakeTensor) or isinstance(value, FakeTensor):
            return False
        return (
            not data.is_mkldnn
            and data.size() == value.size()
            and data.stride() == value.stride()
            and data.dtype == value.dtype
            and data.device == value.device
            and data.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            and data.storage_offset() == value.storage_offset()
        )
    
    utils.is_same_tensor = is_same_tensor
    # We need to do extra-patch because of code like `from xxx import is_same_tensor`
    graph.is_same_tensor = is_same_tensor