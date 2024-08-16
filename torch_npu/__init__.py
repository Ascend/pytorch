# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import builtins
import types
import atexit
import traceback
import warnings

from builtins import isinstance as builtin_isinstance
from typing import Set, Type
from functools import wraps

import torch
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
import torch_npu.distributed
import torch_npu.optim
import torch_npu._C

from torch_npu import profiler
from torch_npu.contrib.function import npu_functional
from torch_npu.contrib.function import fusion_attention
from torch_npu.contrib.module import npu_modules
from torch_npu.utils import apply_module_patch, add_tensor_methods, add_torch_funcs, get_cann_version,\
    serialization_patches, add_storage_methods, add_str_methods, add_dataloader_method, add_asd_patch,\
    add_fx_methods, add_checkpoint_methods, add_launch_methods, path_manager, add_collect_env_methods
from torch_npu.distributed.hccl_dtype_wraper import wrap_dtype_for_hccl
from torch_npu.npu.amp.autocast_mode import apply_autocast_patch
from torch_npu.distributed import fsdp_patches
from torch_npu.utils.exposed_api import public_npu_functions
from torch_npu.utils.error_code import ErrCode, pta_error, _except_handler
from torch_npu.asd.asd import asd_patch
from torch_npu._C._distributed_c10d import ParallelStore
from .version import __version__ as __version__

torch_npu.npu_fusion_attention = fusion_attention.npu_fusion_attention
torch_npu.npu_fusion_attention_grad = fusion_attention.npu_fusion_attention_grad


cann_pytorch_version_map = {
    "6.3.RC2" : ["1.8.1.post2", "1.11.0.post3", "2.0.0.rc1"],
    "6.3.RC1" : ["1.8.1.post1", "1.11.0"],
    "6.1.RC1" : ["1.8.1.post1", "1.11.0"],
    "6.0.1" : ["1.8.1", "1.11.0.rc2"],
    "6.0.RC1" : ["1.8.1", "1.11.0.rc1"]
}


def cann_package_check():
    if "ASCEND_HOME_PATH" in os.environ:
        ascend_home_path = os.environ["ASCEND_HOME_PATH"]
        if not os.path.exists(ascend_home_path):
            raise Exception(f"ASCEND_HOME_PATH : {ascend_home_path} does not exist. " \
                            "Please run 'source set_env.sh' in the CANN installation path.")
        
        # check whether environment variables are correctly configured
        if "ASCEND_OPP_PATH" not in os.environ:
            raise Exception(f"ASCEND_OPP_PATH environment variable is not set. " \
                            "Please check whether the opp package has been installed. If exist, please run "\
                            "'source set_env.sh' in the CANN installation path.")

        ascend_opp_path = os.environ["ASCEND_OPP_PATH"]
        if not os.path.exists(ascend_opp_path):
            raise Exception(f"ASCEND_OPP_PATH : {ascend_opp_path} does not exist. " \
                            "Please check whether the opp package has been installed. If exist, please run "\
                            "'source set_env.sh' in the CANN installation path.")

        ascend_runtime_path = os.path.join(ascend_home_path, "runtime")
        if not os.path.exists(ascend_runtime_path):
            raise Exception(f"ASCEND_RUNTIME_PATH : {ascend_runtime_path} does not exist. " \
                            "Please check whether the runtime package has been installed. If exist, please run "\
                            "'source set_env.sh' in the CANN installation path.")

        ascend_compiler_path = os.path.join(ascend_home_path, "compiler")
        if not os.path.exists(ascend_compiler_path):
            raise Exception(f"ASCEND_COMPILER_PATH : {ascend_compiler_path} does not exist. " \
                            "Please check whether the compiler package has been installed. If exist, please run "\
                            "'source set_env.sh' in the CANN installation path.")

        # get the cann version
        cann_version = get_cann_version()

        # check whether the CANN package version matches the pytorch version
        if cann_version in cann_pytorch_version_map and \
            torch_npu.__version__ not in cann_pytorch_version_map[cann_version]:
            print(f"Warning : CANN package version {cann_version} and PyTorch version {torch_npu.__version__} " \
                  "is not matched, please check the README of the ascend pytorch repo.")
    else:
        print(f"Warning : ASCEND_HOME_PATH environment variable is not set.")

cann_package_check()


_tensor_classes: Set[Type] = set()

NPU_TENSOR = set([
    "FloatTensor", "IntTensor", "DoubleTensor",
    "LongTensor", "ShortTensor", "CharTensor", "ByteTensor", "HalfTensor"])


def _isinstance(obj, class_or_tuple):
    try:
        return builtin_isinstance(obj, class_or_tuple)
    except TypeError as e:
        class_tuple = (class_or_tuple,) if type(class_or_tuple) != tuple else class_or_tuple
        if torch._C.device in class_tuple or torch_npu._C.device in class_tuple:
            return builtin_isinstance(obj, class_tuple + (torch._C.device, torch_npu._C.device))
        raise e


builtins.isinstance = _isinstance

__all__ = []


def wrap_torch_warning_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.warned:
            print(f"Warning: torch.{func.__name__} is deprecated and will be removed in future version. "
                  f"Use torch_npu.{func.__name__} instead.")
            wrapper.warned = True
        return func(*args, **kwargs)
    wrapper.warned = False
    return wrapper

npu_functions = {
    "one_", "fast_gelu", "_amp_foreach_non_finite_check_", "empty_with_format", "unsafe_empty_with_format",
    "copy_memory_", "_dropout_with_byte_mask_backward", "dropout_with_byte_mask", "decode_jpeg", 
    "crop_and_resize", "reverse", "image_normalize", "image_normalize_", "img_to_tensor", 
    "_conv_depthwise2d_backward", "slow_conv_dilated2d_backward", "slow_conv_transpose2d_backward", 
    "batch_norm_reduce", "batch_norm_gather_stats_update", "format_contiguous", "check_match", 
    "check_memory_overlaps", "get_storage_size", "_dropout_with_byte_mask"
}


for name in dir(torch_npu._C._VariableFunctions):
    if name.startswith('__'):
        continue
    globals()[name] = getattr(torch_npu._C._VariableFunctions, name)
    if name in public_npu_functions:
        __all__.append(name)
    if (name in npu_functions) or (name.find("npu") != -1):
        setattr(torch, name, wrap_torch_warning_func(getattr(torch_npu._C._VariableFunctions, name)))
    else:
        setattr(torch, name, getattr(torch_npu._C._VariableFunctions, name))
__all__.append('npu_fusion_attention')

all_monkey_patches = [
    ["npu", torch_npu.npu],
    ["npu.amp", torch_npu.npu.amp],
    ["autograd.profiler", torch_npu.npu.profiler],
    ["distributed", torch_npu.distributed],
    ["nn.parallel.distributed._get_device_index", torch_npu.npu._get_device_index],
    ["distributed.distributed_c10d", torch_npu.distributed.distributed_c10d],
    ["nn.parallel.distributed._get_default_group", torch_npu.distributed.distributed_c10d._get_default_group],
    ["nn.functional", npu_functional],
    ["nn", npu_modules],
    ["device", torch_npu._C.device],
]

all_monkey_patches += serialization_patches
all_monkey_patches += fsdp_patches


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


def apply_class_patches():
    add_storage_methods()
    apply_module_patch()
    add_tensor_methods()
    add_torch_funcs()
    add_str_methods()
    add_dataloader_method()
    wrap_dtype_for_hccl()
    add_fx_methods()
    add_checkpoint_methods()
    apply_autocast_patch()
    add_launch_methods()
    add_collect_env_methods()
    add_asd_patch()


# Apply monkey-patches.
_apply_patches(all_monkey_patches)
apply_class_patches()
asd_patch()
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

torch_npu._C._initExtension()

# Add __doc__ for ops
from . import _op_plugin_docs
del _op_plugin_docs


# NPU exit, need to synchronize devices
def _npu_shutdown():
    success = torch_npu._C._npu_shutdown_synchronize()
    torch_npu.distributed.distributed_c10d._destructor_process_group()
    torch_npu._C._npu_shutdown(success)
    _except_handler.handle_exception()


# register npu shutdown hook on exit
atexit.register(_npu_shutdown)

# Enable NPU Sanitizer
if 'TORCH_NPU_SANITIZER' in os.environ:
    import torch_npu.npu._sanitizer as csan

    csan.enable_npu_sanitizer()
