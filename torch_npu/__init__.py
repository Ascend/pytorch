__all__ = ["erase_stream", "matmul_checksum", "HiFloat8Tensor"]

import os
import sys
import types
import atexit
import traceback
import ctypes
import warnings

from functools import wraps

# Disable autoloading before running 'import torch' to avoid circular dependencies
ORG_AUTOLOAD = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TORCH_WARM_POOL"] = "0"

import torch
from torch.distributed.fsdp import sharded_grad_scaler
from torch.utils.checkpoint import DefaultDeviceType
import torch_npu

acc = torch._C._get_accelerator()
if acc.type != "cpu":
    import time

    # torch_npu.utils._error_code.ErrCode.NOT_SUPPORT
    error_code = "ERR00007"
    error_code_msg = "feature not supported"
    submodule_name = "PTA"
    raise RuntimeError(f"Two accelerators cannot be used at the same time "
                       f"in PyTorch: npu and {acc.type}. You can install "
                       f"the cpu version of PyTorch to use your npu device, "
                       f"or use the {acc.type} device with "
                       f"'export TORCH_DEVICE_BACKEND_AUTOLOAD=0'.\n"
                       f"[ERROR] {time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} "
                       f"(PID:{os.getpid()}, Device:-1, RankID:-1) "
                       f"{error_code} {submodule_name} {error_code_msg}"
                       )

try:
    import torch_npu.npu
except ImportError as e:
    from torch_npu.utils._error_code import ErrCode, pta_error
    if "libhccl.so" in str(e):
        if "ASCEND_OPP_PATH" in os.environ:
            # Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
            e.msg += ". Please check that the compiler package is installed. "\
                       "Please run 'source set_env.sh' in the CANN installation path."\
                        + pta_error(ErrCode.NOT_FOUND)
        else:
            # Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
            e.msg += ". Please check that the cann package is installed. "\
                       "Please run 'source set_env.sh' in the CANN installation path."\
                       + pta_error(ErrCode.NOT_FOUND)
    elif "libascendcl.so" in str(e):
        # Warning: key logs in the fault mode library!!! Don't make arbitrary modifications!!!
        e.msg += ". Please check that the runtime package is installed. "\
                   "Please run 'source set_env.sh' in the CANN installation path."\
                   + pta_error(ErrCode.NOT_FOUND) 
    raise

import torch_npu.npu.amp
import torch_npu.npu.aclnn
import torch_npu.optim
import torch_npu.dynamo
import torch_npu._C
import torch_npu._logging
from torch_npu.utils import patch_getenv
from torch_npu.utils.utils import _is_interactive_command_line
import torch_npu._afd
from torch_npu import profiler
from torch_npu.npu.amp.sharded_grad_scaler import _ShardedGradScaler
from torch_npu.contrib.function import npu_functional
from torch_npu.contrib.module import npu_modules
from torch_npu.utils import _apply_module_patch, _add_tensor_methods, _add_collect_env_methods, \
    _add_storage_methods, _add_serialization_methods, add_dynamo_methods, add_perf_dump_patch, \
    add_optim_method, _inductor_register_device_op_overrides, \
    _apply_npu_show_warning, _apply_npugraph_tree_methods
from torch_npu.utils._dynamo_device import _dynamo_register_interface_for_device
from torch_npu.npu._format import _apply_npu_format_patch
import torch_npu.utils._afd_ops
import torch_npu.utils.custom_ops
import torch_npu.distributed.rpc
import torch_npu.op_plugin
from torch_npu.profiler._add_mstx_patch import _apply_mstx_patch
from torch_npu.distributed.fsdp._add_fsdp_patch import _apply_fsdp_patch
from torch_npu.distributed.rpc.backend_registry import _rpc_backend_registry
from torch_npu.utils import _cann_package_check, _add_intercept_methods
from torch_npu.utils import _register_ops_under_dtensor_rules
from torch_npu.utils.exposed_api import public_npu_functions
from torch_npu.multiprocessing.reductions import _add_reductions_methods
from torch_npu.npu.utils import _erase_stream as erase_stream
from torch_npu.utils.hif8_tensor import _HiFloat8Tensor as HiFloat8Tensor
from torch_npu.utils._error_code import ErrCode, pta_error, _except_handler
from torch_npu.asd.asd import _asd_patch
from torch_npu.asd.checksum import _matmul_checksum as matmul_checksum
from torch_npu._C._distributed_c10d import ParallelStore
from torch_npu.op_plugin.meta import _meta_registrations
from torch_npu.dynamo import _patch_npu_trace_rules
from torch_npu.version import __version__ as __version__
from torch_npu import _op_plugin_docs
del _op_plugin_docs

_cann_package_check()


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

for name in dir(torch_npu._C._cd.DType):
    if name.startswith('__') or name in ['_dir', 'name']:
        continue
    setattr(torch_npu, name, getattr(torch_npu._C._cd.DType, name))

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


def _apply_sharded_grad_scaler_patch():
    torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler = _ShardedGradScaler


def _apply_class_patches():
    _apply_npu_show_warning()
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
    _apply_distributed_methods_patch()
    _apply_mstx_patch()
    _apply_fsdp_patch()
    _apply_npugraph_tree_methods()
    _add_reductions_methods()
    _apply_npu_format_patch()


def _apply_distributed_methods_patch():
    torch._C._distributed_c10d._verify_params_across_processes = torch_npu.distributed._verify_params_across_processes
    torch.distributed.batch_isend_irecv = torch_npu.distributed.distributed_c10d._batch_isend_irecv
    torch.distributed.distributed_c10d.batch_isend_irecv = torch_npu.distributed.distributed_c10d._batch_isend_irecv
    torch.distributed.gather = torch_npu.distributed.distributed_c10d._gather
    torch.distributed.distributed_c10d.gather = torch_npu.distributed.distributed_c10d._gather
    torch.distributed.gather_object = torch_npu.distributed.distributed_c10d._gather_object
    torch.distributed.distributed_c10d.gather_object = torch_npu.distributed.distributed_c10d._gather_object
    torch.distributed.is_hccl_available = torch_npu.distributed.is_hccl_available
    torch.distributed.reinit_process_group = torch_npu.distributed.reinit_process_group
    torch.distributed.distributed_c10d.rendezvous = torch_npu.distributed.distributed_c10d._trigger_rendezvous_decorator(torch.distributed.distributed_c10d.rendezvous)    
    torch.distributed.launcher.api._get_addr_and_port = torch_npu.distributed.distributed_c10d._trigger__get_addr_and_port_decorator(torch.distributed.launcher.api._get_addr_and_port)
    torch._C._distributed_c10d.ProcessGroup._get_sequence_number_for_group = (
        torch_npu.distributed.distributed_c10d._hccl_get_sequence_number_for_group)


torch.utils.rename_privateuse1_backend("npu")
# rename device name to 'npu' and register funcs
torch._register_device_module('npu', torch_npu.npu)
unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True,
                                                     unsupported_dtype=unsupported_dtype)
torch.nn.parameter.UninitializedTensorMixin._allowed_methods.append(torch.Tensor.npu)

# Apply monkey-patches.
_apply_patches(all_monkey_patches)
_apply_class_patches()
_asd_patch()
_except_handler.patch_excepthook()

_warn_msg = {
    "DropoutWithByteMask" : "torch.nn.DropoutWithByteMask is deprecated and will be removed in future version. Use torch_npu.contrib.module.DropoutWithByteMask instead.",
    "dropout_with_byte_mask" : "torch.nn.functional.dropout_with_byte_mask is deprecated and will be removed in future version. Use torch_npu.contrib.function.dropout_with_byte_mask instead.",
}


def _wrap_torch_patch_warning_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(_warn_msg[func.__name__])
        return func(*args, **kwargs)
    return wrapper
setattr(torch.nn, "DropoutWithByteMask", _wrap_torch_patch_warning_func(torch.nn.DropoutWithByteMask))
setattr(torch.nn.functional, "dropout_with_byte_mask", _wrap_torch_patch_warning_func(torch.nn.functional.dropout_with_byte_mask))
# this must be placed at the end
torch_npu._C._initExtension()


def _new_process_group_hccl_helper(dist_backend_opts, pg_options):
    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size
    if pg_options is None or not isinstance(pg_options, torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options):
        pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    pg_options.is_high_priority_stream = False
    pg_options._timeout = dist_backend_opts.timeout
    pg_options.global_ranks_in_group = dist_backend_opts.global_ranks_in_group
    pg_options.group_id = dist_backend_opts.group_id
    return torch_npu._C._distributed_c10d.ProcessGroupHCCL(store, group_rank, group_size, pg_options)


def _new_process_group_lccl_helper(dist_backend_opts, pg_options):
    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size
    return torch_npu._C._distributed_c10d.ProcessGroupLCCL(store, group_rank, group_size)


def _register_distributed_backend_for_npu():
    # init and register lccl backend
    torch.distributed.Backend.register_backend("lccl", lambda dist_backend_opts, pg_options:
        _new_process_group_lccl_helper(dist_backend_opts, pg_options), extended_api=True, devices=["npu"])

    # init and register hccl backend
    # Note: The hccl backend must be registered last. 
    # This is because the "Backend.default_device_backend_map" variable is refreshed during each registration process. 
    # Therefore, it is essential to register the hccl backend last.
    torch.distributed.Backend.register_backend("hccl", lambda dist_backend_opts, pg_options:
        _new_process_group_hccl_helper(dist_backend_opts, pg_options), extended_api=True, devices=["npu"])


# init and register distributed backend
_register_distributed_backend_for_npu()


# set default device type for gradient checkpointing
DefaultDeviceType.set_device_type("npu")
del DefaultDeviceType


# NPU exit, need to synchronize devices
def _npu_shutdown():
    success = torch_npu._C._npu_shutdown_synchronize()
    torch_npu.distributed.distributed_c10d._destructor_process_group()
    torch_npu._C._npu_shutdown(success)
    _except_handler.handle_exception()
    torch_npu.asd.asd.matmul_check._cleanup()
    if torch_npu.npu.aclnn._use_static_aclnn_kernel:
        from torch_npu._inductor.npu_static_kernel import uninstall_static_kernel
        uninstall_static_kernel()


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

# Support stream into Dynamo charts
_patch_npu_trace_rules()

if _is_interactive_command_line():
    os.environ["TASK_QUEUE_ENABLE"] = '0'
    warnings.warn("On the interactive interface, the value of TASK_QUEUE_ENABLE is set to 0 by default. \
                     Do not set it to 1 to prevent some unknown errors")


# This function is an entrypoint called by PyTorch
# when running 'import torch'. There is no need to do anything.
def _autoload():
    # We should restore this switch as sub processes need to inherit its value
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = ORG_AUTOLOAD