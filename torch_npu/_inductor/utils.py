import functools

import torch


# Not good implementation, but no other way
def get_current_raw_stream(device):
    return torch.npu.current_stream(device).npu_stream


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

    from torch._inductor import graph, utils

    utils.is_same_tensor = is_same_tensor
    # We need to do extra-patch because of code like `from xxx import is_same_tensor`
    graph.is_same_tensor = is_same_tensor


def patch_is_gpu():
    from torch._inductor.utils import GPU_TYPES

    GPU_TYPES.append("npu")

    def _return_false(device_interface):
        return False

    torch._inductor.scheduler.device_need_guard = _return_false


def resolve_npu_device_index(device_idx=None) -> int:
    from torch._inductor.utils import decode_device

    return decode_device(torch.device("npu", device_idx)).index


def patch_has_triton():
    from torch._inductor import compile_fx
    from torch_npu.utils._dynamo import has_triton

    torch._inductor.scheduler.has_triton = has_triton
    compile_fx.has_triton = has_triton


def patch_device_supports_tma():
    @functools.lru_cache(None)
    def _device_supports_tma():
        return torch.npu.is_available() and not torch.version.hip

    torch.utils._triton._device_supports_tma = _device_supports_tma


def disable_foreach():
    from torch._inductor.scheduler import Scheduler

    def create_foreach_nodes(self):
        return

    Scheduler.create_foreach_nodes = create_foreach_nodes
