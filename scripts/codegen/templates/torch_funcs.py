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

import numpy
import torch
import torch_npu
from torch_npu.utils.device_guard import torch_device_guard, device


@torch_device_guard
def _device(*args, **kwargs):
    new_device = torch_npu.new_device(*args, **kwargs)
    if new_device.type == torch.npu.native_device:
        return device(type=torch_npu.npu.npu_device, index=new_device.index)
    return new_device


@torch_device_guard
def _tensor(*args, **kwargs):
    return torch_npu.tensor(*args, **kwargs)


@torch_device_guard
def _full(*args, **kwargs):
    return torch_npu.full(*args, **kwargs)


@torch_device_guard
def _randint(*args, **kwargs):
    return torch_npu.randint(*args, **kwargs)


@torch_device_guard
def _range(*args, **kwargs):
    return torch_npu.range(*args, **kwargs)


@torch_device_guard
def _arange(*args, **kwargs):
    return torch_npu.arange(*args, **kwargs)


@torch_device_guard
def _empty_with_format(*args, **kwargs):
    return torch_npu.empty_with_format(*args, **kwargs)


@torch_device_guard
def _npu_dropout_gen_mask(*args, **kwargs):
    return torch_npu.npu_dropout_gen_mask(*args, **kwargs)


def jit_script(obj, optimize=None, _frames_up=0, _rcb=None):
    # (Ascend) Disable extension of torch.jit.script
    return obj


def _as_tensor(*args, **kwargs):
    if kwargs and "device" in kwargs:
        dst_device = kwargs.get("device")
        if "npu" in str(dst_device):
            if isinstance(args[0], numpy.ndarray):
                return torch.from_numpy(args[0]).to(dst_device)
            else:
                return args[0].to(dst_device)
    return torch._C._VariableFunctions.as_tensor(*args, **kwargs)


${device_methods_def_py_dispatch}

def add_torch_funcs():
    torch.device = _device
    torch.tensor = _tensor
    torch.full = _full
    torch.randint = _randint
    torch.range = _range
    torch.arange = _arange
    torch.empty_with_format = _empty_with_format
    torch.npu_dropout_gen_mask = _npu_dropout_gen_mask
    torch.jit.script = jit_script
    torch.as_tensor = _as_tensor

${device_methods_def_py}