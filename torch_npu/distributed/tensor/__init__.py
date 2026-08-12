import importlib

import torch_npu.distributed.tensor._dtensor_patch  # patch before register strategy
import torch_npu.distributed.tensor._attention
import torch_npu.distributed.tensor._math_ops
import torch_npu.distributed.tensor._matrix_ops
import torch_npu.distributed.tensor._moe_ops
import torch_npu.distributed.tensor._pointwise_ops
import torch_npu.distributed.tensor._sharded_tensor_patch
import torch_npu.distributed.tensor._view_ops


def __getattr__(name):
    if name != "experimental":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.experimental")
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | {"experimental"})
