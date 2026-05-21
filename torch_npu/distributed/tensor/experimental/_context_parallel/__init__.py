import functools

import torch_npu.distributed.tensor.experimental._context_parallel._attention
from torch.distributed.tensor.experimental._attention import (
    _AttentionContextParallel,
    context_parallel as _torch_context_parallel,
    context_parallel_unshard as _torch_context_parallel_unshard,
    set_rotate_method as _torch_set_rotate_method,
)

__all__ = [
    "_AttentionContextParallel",
    "context_parallel",
    "context_parallel_unshard",
    "set_rotate_method",
]

_WRAPPER_ASSIGNMENTS = ("__name__", "__qualname__", "__doc__", "__annotations__")


@functools.wraps(_torch_context_parallel, assigned=_WRAPPER_ASSIGNMENTS)
def context_parallel(*args, **kwargs):
    return _torch_context_parallel(*args, **kwargs)


@functools.wraps(_torch_context_parallel_unshard, assigned=_WRAPPER_ASSIGNMENTS)
def context_parallel_unshard(*args, **kwargs):
    return _torch_context_parallel_unshard(*args, **kwargs)


@functools.wraps(_torch_set_rotate_method, assigned=_WRAPPER_ASSIGNMENTS)
def set_rotate_method(*args, **kwargs):
    return _torch_set_rotate_method(*args, **kwargs)
