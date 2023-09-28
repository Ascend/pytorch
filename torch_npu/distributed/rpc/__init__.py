import torch.distributed.rpc as rpc


def is_available() -> bool:
    return rpc.is_available()

if is_available():
    from torch_npu._C._distributed_rpc import (
        TensorPipeAgent
    ) 
    
    from . import backend_registry
