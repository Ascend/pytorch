import torch.distributed.rpc as rpc

__all__ = []


if rpc.is_available():
    from . import backend_registry
