import functools
import torch
import torch_npu


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
    
    from torch._inductor import utils, graph
    utils.is_same_tensor = is_same_tensor
    # We need to do extra-patch because of code like `from xxx import is_same_tensor`
    graph.is_same_tensor = is_same_tensor


def patch_is_gpu():
    from torch._inductor.utils import GPU_TYPES
    GPU_TYPES.append('npu')


def patch_has_triton():
    from torch.utils._triton import has_triton_package

    @functools.lru_cache(None)
    def has_triton() -> bool:
        if not has_triton_package():
            return False

        from torch._dynamo.device_interface import get_interface_for_device

        def cuda_extra_check(device_interface):
            return True

        def cpu_extra_check(device_interface):
            import triton.backends

            return "cpu" in triton.backends.backends

        def _return_true(device_interface):
            return True

        triton_supported_devices = {
            "cuda": cuda_extra_check,
            "xpu": _return_true,
            "cpu": cpu_extra_check,
            "npu": _return_true
        }

        def is_device_compatible_with_triton():
            for device, extra_check in triton_supported_devices.items():
                device_interface = get_interface_for_device(device)
                if device_interface.is_available() and extra_check(device_interface):
                    return True
            return False

        return is_device_compatible_with_triton()

    torch.utils._triton.has_triton = has_triton
    torch._inductor.scheduler.has_triton = has_triton


