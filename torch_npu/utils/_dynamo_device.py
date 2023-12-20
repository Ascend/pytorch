import torch
from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device, \
    caching_worker_current_devices, caching_worker_device_properties
from ._device import NPUDevice
from ..npu.streams import Event, Stream
from ..npu.utils import current_device, set_device, device_count, stream, current_stream, \
    set_stream, synchronize, get_device_capability
from ..npu.utils import get_device_properties as get_device_properties_npu


class NpuInterface(DeviceInterface):
    device = NPUDevice
    torch.device = NPUDevice
    Event = Event
    Stream = Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["npu"] = device

        @staticmethod
        def current_device() -> int:
            if "npu" in caching_worker_current_devices:
                return caching_worker_current_devices["npu"]
            return current_device()

        @staticmethod
        def get_device_properties(device=None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != "npu":
                        raise AssertionError('device.type should be equal to npu.')
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = NpuInterface.Worker.current_device()

            if "npu" not in caching_worker_device_properties:
                device_prop = [
                    get_device_properties_npu(i)
                    for i in range(device_count())
                ]
                caching_worker_device_properties["npu"] = device_prop

            return caching_worker_device_properties["npu"][device]

    current_device = staticmethod(current_device)
    set_device = staticmethod(set_device)
    device_count = staticmethod(device_count)
    stream = staticmethod(stream)
    current_stream = staticmethod(current_stream)
    set_stream = staticmethod(set_stream)
    synchronize = staticmethod(synchronize)
    get_device_properties = staticmethod(get_device_properties_npu)

    # Currently, NPU does not support _set_stream_by_id and get_raw_stream.
    _set_stream_by_id = staticmethod(None)
    get_raw_stream = staticmethod(None)

    @staticmethod
    def is_available() -> bool:
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        major, minor = get_device_capability(device)
        return major * 10 + minor


def _dynamo_register_interface_for_device():
    register_interface_for_device("npu", NpuInterface)
