import functools

from torch._inductor.runtime.hints import DeviceProperties


def patch_create_device_properties():
    DeviceProperties.create = NPUDeviceProperties.create


class NPUDeviceProperties(DeviceProperties):

    @classmethod
    @functools.lru_cache(None)
    def create(cls, device) -> DeviceProperties:
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        props = device_interface.get_device_properties(device)

        try:
            multi_processor_count = props.vector_core_num
        except AttributeError:
            if device_type == "xpu":
                multi_processor_count = props.gpu_subslice_count
            else:
                raise
        return DeviceProperties(
            type=device_type,
            index=device.index,
            multi_processor_count=multi_processor_count,
            cc=device_interface.get_compute_capability(device),
            major=getattr(props, "major", None),
            regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
            max_threads_per_multi_processor=getattr(
                props, "max_threads_per_multi_processor", None
            ),
            warp_size=getattr(props, "warp_size", 32 if device_type != "cpu" else None),
        )
