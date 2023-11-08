import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Type

import torch
import torch.distributed as dist
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook

from torch.utils._pytree import tree_flatten, tree_unflatten

RPC_AVAILABLE = False
if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
    from torch.distributed.utils import (
        _alloc_storage,
        _cast_forward_inputs,
        _free_storage,
        _sync_module_states,
        _to_kwargs,
        _verify_param_shape_across_processes,
    )
if torch.distributed.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef

from torch._utils import _get_device_index
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules import Module
from torch.nn.parallel.distributed import _MixedPrecision
import torch_npu


class Distributed_DataParallel(DistributedDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.npu.synchronize()