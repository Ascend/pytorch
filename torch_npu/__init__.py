import os
import sys
import types
import atexit
import traceback
import ctypes

from functools import wraps

import torch
from torch.distributed.fsdp import sharded_grad_scaler
from torch.utils.checkpoint import DefaultDeviceType
import torch_npu

try:
    import torch_npu.npu
except ImportError as e:
    from torch_npu.utils.error_code import ErrCode, pta_error
    if "libhccl.so" in str(e):
        if "ASCEND_OPP_PATH" in os.environ:
            e.msg += ". Please check that the compiler package is installed. "\
                       "Please run 'source set_env.sh' in the CANN installation path."\
                        + pta_error(ErrCode.NOT_FOUND)
        else:
            e.msg += ". Please check that the cann package is installed. "\
                       "Please run 'source set_env.sh' in the CANN installation path."\
                       + pta_error(ErrCode.NOT_FOUND)
    elif "libascendcl.so" in str(e):
        e.msg += ". Please check that the runtime package is installed. "\
                   "Please run 'source set_env.sh' in the CANN installation path."\
                   + pta_error(ErrCode.NOT_FOUND) 
    raise

import torch_npu.npu.amp
import torch_npu.npu.aclnn
import torch_npu.optim
import torch_npu.dynamo
import torch_npu._C
from torch_npu import profiler
from torch_npu.contrib.function import npu_functional
from torch_npu.contrib.module import npu_modules
from torch_npu.utils import _apply_module_patch, _add_tensor_methods, _add_collect_env_methods, \
    _add_storage_methods, _add_serialization_methods, add_dynamo_methods, add_perf_dump_patch, \
    _dynamo_register_interface_for_device, add_optim_method, _inductor_register_device_op_overrides
import torch_npu.utils.custom_ops
import torch_npu.distributed.rpc
from torch_npu.distributed.rpc.backend_registry import _rpc_backend_registry
from torch_npu.utils import _cann_package_check, _add_intercept_methods
from torch_npu.utils import _register_ops_under_dtensor_rules
from torch_npu.utils.exposed_api import public_npu_functions
from torch_npu.utils.error_code import ErrCode, pta_error, _except_handler
from torch_npu.asd.asd import _asd_patch
from .version import __version__ as __version__
from .meta import _meta_registrations
from . import _op_plugin_docs
del _op_plugin_docs


_cann_package_check()


__all__ = ["ErrCode"]


def _wrap_torch_error_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"torch.{func.__name__} is deprecated and will be removed in future version. "
                           f"Use torch_npu.{func.__name__} instead." + pta_error(ErrCode.NOT_SUPPORT))

    return wrapper


for name in dir(torch.ops.npu):
    if name.startswith('__') or name in ['_dir', 'name']:
        continue
    globals()[name] = getattr(torch.ops.npu, name)
    if name in public_npu_functions:
        __all__.append(name)
    setattr(torch, name, _wrap_torch_error_func(getattr(torch.ops.npu, name)))

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
            raise NotImplementedError("Patch module must have __all__ definition." + pta_error(ErrCode.NOT_SUPPORT))
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))


def _apply_distributed_patches():
    _apply_patches([["distributed", torch_npu.distributed]])


def _apply_sharded_grad_scaler_patch():
    torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler = torch_npu.npu.amp.ShardedGradScaler


def _apply_class_patches():
    _add_storage_methods()
    _apply_module_patch()
    _add_tensor_methods()
    _add_serialization_methods()
    _add_intercept_methods()
    _add_collect_env_methods()
    add_dynamo_methods()
    add_optim_method()
    _apply_sharded_grad_scaler_patch()
    add_perf_dump_patch()


def _try_preload_opapi():
    try:
        ctypes.CDLL("libopapi.so")
    except OSError as exception:
        exception.args = (exception.args[0] + ". Please check that the opp package is installed. "\
                "Please run 'source set_env.sh' in the CANN installation path." + pta_error(ErrCode.NOT_FOUND),)
        raise

torch.utils.rename_privateuse1_backend("npu")
# rename device name to 'npu' and register funcs
torch._register_device_module('npu', torch_npu.npu)
unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True,
                                                     unsupported_dtype=unsupported_dtype)

# Apply monkey-patches.
_apply_patches(all_monkey_patches)
_apply_distributed_patches()
_apply_class_patches()
_asd_patch()
_except_handler.patch_excepthook()
_try_preload_opapi()
# this must be placed at the end
torch_npu._C._initExtension()


def _new_process_group_hccl_helper(dist_backend_opts, pg_options):
    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size
    pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    pg_options.is_high_priority_stream = False
    pg_options._timeout = dist_backend_opts.timeout
    pg_options.global_ranks_in_group = dist_backend_opts.global_ranks_in_group
    return torch_npu._C._distributed_c10d.ProcessGroupHCCL(store, group_rank, group_size, pg_options)


# init and register hccl backend
torch.distributed.Backend.register_backend("hccl", lambda dist_backend_opts, pg_options:
    _new_process_group_hccl_helper(dist_backend_opts, pg_options), extended_api=True, devices=["npu"])


# set default device type for gradient checkpointing
DefaultDeviceType.set_device_type("npu")
del DefaultDeviceType


# NPU exit, need to synchronize devices
def _npu_shutdown():
    torch_npu._C._npu_shutdown()
    _except_handler.handle_exception()


# register npu shutdown hook on exit
atexit.register(_npu_shutdown)

# init and register rpc npu backend
_rpc_backend_registry()

# register rules for ops in dtensor
_register_ops_under_dtensor_rules()

# register npu device interface for dynamo
_dynamo_register_interface_for_device()

# Enable NPU Sanitizer
if 'TORCH_NPU_SANITIZER' in os.environ:
    import torch_npu.npu._sanitizer as csan

    csan.enable_npu_sanitizer()

# register npu device op overrides for inductor
_inductor_register_device_op_overrides()
