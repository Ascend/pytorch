# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import functools
from importlib import import_module

# Import NPU _attention module to trigger side-effect injection (replaces native dispatcher functions)
import_module("torch_npu.distributed.tensor.experimental._context_parallel._attention")

# Re-export public API from native module (_enable_cp_dtensor_dispatcher is now the NPU-injected version)
from torch.distributed.tensor.experimental._context_parallel._attention import (
    _CausalBehavior,
    _ContextParallel,
    _context_parallel_shard,
    _cp_options,
    _disable_context_parallel_dispatcher,
    _enable_context_parallel_dispatcher,
    _is_causal_behavior,
    _RotateMethod,
    context_parallel as _torch_context_parallel,
    context_parallel_unshard as _torch_context_parallel_unshard,
    set_rotate_method as _torch_set_rotate_method,
)
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
)

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


__all__ = [
    # From native _context_parallel._attention
    "_CausalBehavior",
    "_context_parallel_shard",
    "_ContextParallel",
    "_cp_options",
    "_disable_context_parallel_dispatcher",
    "_enable_context_parallel_dispatcher",
    "_is_causal_behavior",
    "_RotateMethod",
    "context_parallel",
    "context_parallel_unshard",
    "set_rotate_method",
    # From native _context_parallel._load_balancer
    "_HeadTailLoadBalancer",
    "_LoadBalancer",
    "_PerDocumentHeadTailLoadBalancer",
]
