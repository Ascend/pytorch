import torch
from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device, \
    caching_worker_current_devices, caching_worker_device_properties

from torch_npu._C import _npu_getCurrentRawStream as get_npu_stream
from ..npu.streams import Event, Stream
from ..npu import current_device, set_device, device_count, stream, current_stream, \
    set_stream, synchronize, get_device_capability
from ..npu import get_device_properties as get_device_properties_npu


class NpuInterface(DeviceInterface):
    device = torch.device
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

    # Currently, NPU does not support _set_stream_by_id.
    _set_stream_by_id = staticmethod(None)
    get_raw_stream = staticmethod(get_npu_stream)

    @staticmethod
    def is_available() -> bool:
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        r"""Different from cuda, only return the chip model here.
        """
        return torch.npu.get_device_name(device)

    @staticmethod
    def exchange_device(device: int) -> int:
        curr_device = current_device()
        set_device(device)
        return curr_device

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return device

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False):
        return True


def _dynamo_register_interface_for_device():
    register_interface_for_device("npu", NpuInterface)
    for i in range(16):
        register_interface_for_device(f"npu:{i}", NpuInterface)
