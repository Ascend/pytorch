import torch
from torch import device as origin_device
from torch.fx.graph import _register_custom_builtin


class MetaDevice(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, origin_device)

    def __eq__(self, other):
        return other == origin_device

    def __ne__(self, other):
        return other != origin_device

    def __hash__(self):
        return hash(origin_device)

    def __format__(self, format_spec):
        return f"{origin_device}"


class NPUDevice(metaclass=MetaDevice):
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], int):
            args = ("npu", args[0])
        elif isinstance(kwargs.get('device'), int):
            kwargs["device"] = f"npu:{kwargs.get('device')}"
        return origin_device(*args, **kwargs)


def apply_device_patch():
    torch.device = NPUDevice
    _register_custom_builtin('device', 'from torch import device', NPUDevice)
