import os
import re
import sys
import types
import atexit
import traceback

from typing import Set, Type
from functools import wraps

import torch
from . import _ld_preload  # noqa: F401
import torch_npu

try:
    import torch_npu.npu
except ImportError as e:
    if "libhccl.so" in str(e):
        ei = sys.exc_info()
        if ("ASCEND_OPP_PATH" in os.environ):
            newErr = ImportError(str(ei[1]) + ". Please check that the compiler package is installed. " \
                                 "Please run 'source set_env.sh' in the CANN installation path.")
        else :
            newErr = ImportError(str(ei[1]) + ". Please check that the cann package is installed. " \
                                 "Please run 'source set_env.sh' in the CANN installation path.")
        traceback.print_exception(ei[0], newErr, ei[2])
        sys.exit()

    if "libascendcl.so" in str(e):
        ei = sys.exc_info()
        newErr = ImportError(str(ei[1]) + ". Please check that the runtime package is installed. " \
                             "Please run 'source set_env.sh' in the CANN installation path.")
        traceback.print_exception(ei[0], newErr, ei[2])
        sys.exit()

    else:
        traceback.print_exc()
import torch_npu.npu.amp
import torch_npu.npu.aclnn
import torch_npu._C
import torch_npu.npu.npu_print as _npu_print

from torch_npu.contrib.function import npu_functional
from torch_npu.contrib.module import npu_modules
from torch_npu.utils import apply_module_patch, add_tensor_methods, \
     add_storage_methods
import torch_npu.utils.custom_ops
from torch_npu.npu.profiler import add_profiler_methods
from .version import __version__ as __version__


cann_pytorch_version_map = {
    "6.3.RC2" : ["1.8.1.post2", "1.11.0.post1", "2.0.0.rc1"],
    "6.3.RC1" : ["1.8.1.post1", "1.11.0"],
    "6.1.RC1" : ["1.8.1.post1", "1.11.0"],
    "6.0.1" : ["1.8.1", "1.11.0.rc2"],
    "6.0.RC1" : ["1.8.1", "1.11.0.rc1"]
}

def get_cann_version(ascend_home_path):
    cann_version = ""
    for dirpath, _, filenames in os.walk(os.path.realpath(ascend_home_path)):
        if cann_version:
            break
        install_files = [file for file in filenames if re.match(r"ascend_.*_install\.info", file)]
        if install_files:
            filepath = os.path.join(dirpath, install_files[0])
            with open(filepath, "r") as f:
                for line in f:
                    if line.find("version") != -1:
                        cann_version = line.strip().split("=")[-1]
                        break
    return cann_version

def cann_package_check():
    if "ASCEND_HOME_PATH" in os.environ:
        ascend_home_path = os.environ["ASCEND_HOME_PATH"]
        if not os.path.exists(ascend_home_path):
            raise Exception(f"ASCEND_HOME_PATH : {ascend_home_path} does not exist. " \
                            "Please run 'source set_env.sh' in the CANN installation path.")
        
        # check whether environment variables are correctly configured
        if "ASCEND_OPP_PATH" not in os.environ:
            raise Exception(f"ASCEND_OPP_PATH environment variable is not set. " \
                            "Please run 'source set_env.sh' in the CANN installation path.")

        ascend_opp_path = os.environ["ASCEND_OPP_PATH"]
        if not os.path.exists(ascend_opp_path):
            raise Exception(f"ASCEND_OPP_PATH : {ascend_opp_path} does not exist. " \
                            "Please run 'source set_env.sh' in the CANN installation path.")

        ascend_runtime_path = os.path.join(ascend_home_path, "runtime")
        if not os.path.exists(ascend_runtime_path):
            raise Exception(f"{ascend_runtime_path} does not exist. " \
                            "Please run 'source set_env.sh' in the CANN installation path.")

        ascend_compiler_path = os.path.join(ascend_home_path, "compiler")
        if not os.path.exists(ascend_compiler_path):
            raise Exception(f"{ascend_compiler_path} does not exist. " \
                            "Please run 'source set_env.sh' in the CANN installation path.")

        # get the cann version
        cann_version = get_cann_version(ascend_home_path)

        # check whether the CANN package version matches the pytorch version
        if cann_version in cann_pytorch_version_map and \
            torch_npu.__version__ not in cann_pytorch_version_map[cann_version]:
            print(f"Warning : CANN package version {cann_version} and PyTorch version {torch_npu.__version__} " \
                  "is not matched, please check the README in repo of https://gitee.com/ascend/pytorch")
    else:
        print(f"Warning : ASCEND_HOME_PATH environment variable is not set.")

cann_package_check()

graph_printer = _npu_print.GraphPrinter()

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

        assert hasattr(patch, '__all__'), "Patch module must have __all__ definition."
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))


def apply_class_patches():
    add_storage_methods()
    apply_module_patch()
    add_tensor_methods()
    add_profiler_methods()

# rename device name to 'npu' and register funcs
torch._register_device_module('npu', torch_npu.npu)
unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True,
                                                     unsupported_dtype=unsupported_dtype)

# Apply monkey-patches.
_apply_patches(all_monkey_patches)
apply_class_patches()
torch.distributed.is_hccl_available = lambda : True

# this must be placed at the end
torch_npu._C._initExtension()

# init and register hccl backend
torch_npu._C._c10d_npu_init()
torch.distributed.Backend.register_backend("hccl", lambda store, group_rank, group_size, timeout :
    torch_npu._C._distributed_c10d.ProcessGroupHCCL(store, group_rank, group_size, timeout), devices=["npu"])

# set default device type for gradient checkpointing
from torch.utils.checkpoint import DefaultDeviceType
DefaultDeviceType.set_device_type("npu")
del DefaultDeviceType

# NPU exit, need to synchronize devices
def _npu_shutdown():
    torch_npu._C._npu_shutdown()

#register npu shutdown hook on exit
atexit.register(_npu_shutdown)
