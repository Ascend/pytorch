import torch
from torch.overrides import TorchFunctionMode
from torch.utils._device import _device_constructors


class NPUDeviceContext(TorchFunctionMode):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.device and args and isinstance(args[0], int):
            args = ("npu", args[0])
        elif (func in _device_constructors() or func == torch.device) and isinstance(kwargs.get('device'), int):
            kwargs["device"] = f"npu:{kwargs.get('device')}"
        return func(*args, **kwargs)
