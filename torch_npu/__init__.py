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
import inspect
import types
import atexit
import traceback

from builtins import isinstance as builtin_isinstance
from typing import Set, Type
from functools import wraps

import torch
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
import torch_npu.distributed
import torch_npu.optim
import torch_npu._C

import torch_npu.npu.npu_print as _npu_print
from torch_npu import profiler
from torch_npu.contrib.function import npu_functional
from torch_npu.contrib.module import npu_modules
from torch_npu.utils import apply_module_patch, add_tensor_methods, add_torch_funcs, \
    serialization_patches, add_storage_methods, add_str_methods, add_dataloader_method, \
    add_fx_methods, add_checkpoint_methods
from torch_npu.distributed.hccl_dtype_wraper import wrap_dtype_for_hccl
from torch_npu.npu.amp.autocast_mode import apply_autocast_patch
from torch_npu.distributed import fsdp_patches

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

_tensor_classes: Set[Type] = set()

NPU_TENSOR = set([
    "FloatTensor", "IntTensor", "DoubleTensor",
    "LongTensor", "ShortTensor", "CharTensor", "ByteTensor", "HalfTensor"])


def _isinstance(obj, class_or_tuple):
    try:
        return builtin_isinstance(obj, class_or_tuple)
    except TypeError as e:
        class_tuple = (class_or_tuple,) if type(class_or_tuple) != tuple else class_or_tuple
        if hasattr(obj, "type") and callable(obj.type) and inspect.getfullargspec(obj.type).args == ['self']:
            type_str = str(obj.type())
            tensor_type = type_str.split('.')[-1]
            if f"npu.{tensor_type}" in type_str and tensor_type in NPU_TENSOR:
                return eval(type_str) in class_tuple

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
    "empty_with_format", "copy_memory_", "_dropout_with_byte_mask_backward", "dropout_with_byte_mask", 
    "decode_jpeg", "crop_and_resize", "reverse", "image_normalize", "image_normalize_", "img_to_tensor", 
    "_conv_depthwise2d_backward", "slow_conv_dilated2d_backward", "slow_conv_transpose2d_backward", 
    "batch_norm_reduce", "batch_norm_gather_stats_update", "format_contiguous", "check_match", 
    "check_memory_overlaps", "get_storage_size", "bscpp_add", "_dropout_with_byte_mask", "empty_with_format"
}



for name in dir(torch_npu._C._VariableFunctions):
    if name.startswith('__'):
        continue
    globals()[name] = getattr(torch_npu._C._VariableFunctions, name)
    __all__.append(name)
    if (name in npu_functions) or (name.find("npu") != -1):
        setattr(torch, name, wrap_torch_warning_func(getattr(torch_npu._C._VariableFunctions, name)))
    else:
        setattr(torch, name, getattr(torch_npu._C._VariableFunctions, name))

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
    ["_C.Generator", torch_npu._C.Generator],
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

        assert hasattr(patch, '__all__'), "Patch module must have __all__ definition."
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


# Apply monkey-patches.
_apply_patches(all_monkey_patches)
apply_class_patches()
torch_npu._C._initExtension()


# NPU exit, need to synchronize devices
def _npu_shutdown():
    torch_npu._C._npu_shutdown()


# register npu shutdown hook on exit
atexit.register(_npu_shutdown)
