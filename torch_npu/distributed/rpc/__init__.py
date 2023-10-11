import torch.distributed.rpc as rpc


def is_available() -> bool:
    return rpc.is_available()

if is_available():
    from . import backend_registry
