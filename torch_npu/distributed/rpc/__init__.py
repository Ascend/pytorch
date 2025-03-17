__all__ = []

import torch.distributed.rpc as rpc


if rpc.is_available():
    from . import backend_registry
