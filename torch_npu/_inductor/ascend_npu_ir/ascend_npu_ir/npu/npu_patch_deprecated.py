from typing import Dict,Any
import hashlib
import functools
import json

import torch
from torch._inductor.codecache import CacheBase
from torch.distributed import distributed_c10d
from torch.distributed.distributed_c10d import (
    _world,
    timedelta,
)
from torch.library import Library, impl
from ..npu.utils import get_anir_mode

python_dispatcher_lib = Library("aten", "IMPL", "PythonDispatcher")

@impl(python_dispatcher_lib, "embedding_backward")
def embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    if sparse != False:
        raise RuntimeError("the current NPU does not yet support sparse tensor, when sparse is set to True")
    return torch.ops.aten.embedding_dense_backward(grad, indices,  num_weights, padding_idx, scale_grad_by_freq)

@impl(python_dispatcher_lib, "contiguous")
def py_contiguous(x, memory_format=torch.contiguous_format):
    return x.clone(memory_format=memory_format)


@staticmethod
@functools.lru_cache(None)
def _patch_get_system() -> Dict[str, Any]:
    system = {}
    system["hash"] = hashlib.sha256(
        json.dumps(system, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return system

def _patch_add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
    """
    This API adds an ephemeral timeout extension for all PGs locally
    on one rank. The timeout gets reset when the first collective issued
    after API called finished.
    NOTE: We only support to set timeout for cuda backends for now.
    NOTE: While this feature
    provides flexibility in specific scenarios, it introduces statefulness
    to timeout setting. Therefore, it is advisable to use this API sparingly
    and consider alternative approaches, such as directly setting the timeout
    or utilizing a barrier collective (one can set any timeout to the barrier),
    whenever feasible.

    Args:
        timeout (timedelta): The delta of timeout to extend.

    Returns:
        None.
    """
    for pg in _world.pg_map.keys():
        devices = pg._device_types
        if torch.device("npu") in devices:
            backend = pg._get_backend(torch.device("npu"))
                
CacheBase.get_system = _patch_get_system
distributed_c10d._add_ephemeral_timeout_for_all_pgs = _patch_add_ephemeral_timeout_for_all_pgs

if get_anir_mode() == 'O0':
    @torch.ops.aten.silu_backward.default.py_impl(torch._C.DispatchKey.CompositeImplicitAutograd)
    def my_silu_backward(grad_out, self):
        # use with some caution: this is only really valid to run in the context of proxy tensor tracing
        return NotImplemented
    
