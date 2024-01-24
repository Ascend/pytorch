import os
import sys
import types
import atexit
import traceback

from functools import wraps

import torch
from torch.utils.checkpoint import DefaultDeviceType
import torch_npu

try:
    import torch_npu.npu
except ImportError as e:
    if "libhccl.so" in str(e):
        ei = sys.exc_info()
        if "ASCEND_OPP_PATH" in os.environ:
            newErr = ImportError(str(ei[1]) + ". Please check that the compiler package is installed. "
                                 "Please run 'source set_env.sh' in the CANN installation path.")
        else:
            newErr = ImportError(str(ei[1]) + ". Please check that the cann package is installed. "
                                 "Please run 'source set_env.sh' in the CANN installation path.")
        traceback.print_exception(ei[0], newErr, ei[2])
        sys.exit()

    if "libascendcl.so" in str(e):
        ei = sys.exc_info()
        newErr = ImportError(str(ei[1]) + ". Please check that the runtime package is installed. "
                             "Please run 'source set_env.sh' in the CANN installation path.")
        traceback.print_exception(ei[0], newErr, ei[2])
        sys.exit()

    else:
        traceback.print_exc()

import torch_npu.npu.amp
import torch_npu.npu.aclnn
import torch_npu.optim
import torch_npu.dynamo
import torch_npu._C
from torch_npu import profiler
from torch_npu.contrib.function import npu_functional
from torch_npu.contrib.module import npu_modules
from torch_npu.utils import apply_module_patch, add_tensor_methods, add_collect_env_methods,\
    add_storage_methods, add_serialization_methods, apply_device_patch, add_dynamo_methods,\
    _dynamo_register_interface_for_device, add_optim_method
import torch_npu.utils.custom_ops
import torch_npu.distributed.rpc
from torch_npu.distributed.rpc.backend_registry import rpc_backend_registry
from torch_npu.utils import cann_package_check, add_intercept_methods
from torch_npu.utils import register_ops_under_dtensor_rules
from torch_npu.utils.exposed_api import public_npu_functions
from torch_npu.distributed.optim.zero_redundancy_optimizer import _get_optimizer_constructor
from .version import __version__ as __version__
from .meta import meta_registrations

cann_package_check()


__all__ = []


def wrap_torch_error_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"torch.{func.__name__} is deprecated and will be removed in future version. "
                           f"Use torch_npu.{func.__name__} instead.")
    return wrapper


for name in dir(torch.ops.npu):
    if name.startswith('__') or name in ['_dir', 'name']:
        continue
    globals()[name] = getattr(torch.ops.npu, name)
    if name in public_npu_functions:
        __all__.append(name)
    setattr(torch, name, wrap_torch_error_func(getattr(torch.ops.npu, name)))

all_monkey_patches = [
    ["nn.functional", npu_functional],
    ["nn", npu_modules],
]


def _apply_patches(monkey_patches):

    def _getattr(module_list, root_module=torch):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))
        else:
            empty_module_name = f'{root_module.__name__}.{module_list[0]}'
            sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
            setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for patch_pair in monkey_patches:
        dest, patch = patch_pair
        dest_module = _getattr(dest.split('.'), root_module=torch)
        last_module_level = dest.split(".")[-1]
        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, '__all__'):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f'{dest_module.__name__}.{last_module_level}'] = patch
            continue

        if not hasattr(patch, '__all__'):
            raise NotImplementedError("Patch module must have __all__ definition.")
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))


def _apply_distributed_patches():
    torch.nn.parallel.DistributedDataParallel._ddp_init_helper = torch_npu.utils.module._ddp_init_helper
    _apply_patches([["distributed", torch_npu.distributed]])


def apply_zero_patch():
    torch.distributed.optim.ZeroRedundancyOptimizer._get_optimizer_constructor = _get_optimizer_constructor


def apply_class_patches():
    add_storage_methods()
    apply_module_patch()
    apply_device_patch()
    add_tensor_methods()
    add_serialization_methods()
    add_intercept_methods()
    add_collect_env_methods()
    add_dynamo_methods()
    add_optim_method()
    apply_zero_patch()



torch.utils.rename_privateuse1_backend("npu")
# rename device name to 'npu' and register funcs
torch._register_device_module('npu', torch_npu.npu)
unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True,
                                                     unsupported_dtype=unsupported_dtype)

# Apply monkey-patches.
_apply_patches(all_monkey_patches)
_apply_distributed_patches()
apply_class_patches()
torch.distributed.is_hccl_available = lambda : True

# this must be placed at the end
torch_npu._C._initExtension()

# init and register hccl backend
torch.distributed.Backend.register_backend("hccl", lambda store, group_rank, group_size, timeout:
    torch_npu._C._distributed_c10d.ProcessGroupHCCL(store, group_rank, group_size, timeout), devices=["npu"])

#PGHCCL batch_isend_irecv
torch.distributed.batch_isend_irecv = torch_npu.distributed.batch_isend_irecv

# set default device type for gradient checkpointing
DefaultDeviceType.set_device_type("npu")
del DefaultDeviceType


# NPU exit, need to synchronize devices
def _npu_shutdown():
    torch_npu._C._npu_shutdown()


# register npu shutdown hook on exit
atexit.register(_npu_shutdown)


# init and register rpc npu backend
rpc_backend_registry()

torch._dynamo.skipfiles.add(torch_npu.utils._device)

# register rules for ops in dtensor
register_ops_under_dtensor_rules()

# register npu device interface for dynamo
_dynamo_register_interface_for_device()

