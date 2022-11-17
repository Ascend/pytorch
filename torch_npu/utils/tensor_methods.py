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


import warnings
import numpy
import torch

import torch_npu
from torch_npu.utils.device_guard import torch_device_guard, device
from .storage import _reduce_ex


warnings.filterwarnings(action="once")
warning_str = "The tensor methods of custom operators would cause performance drop." + \
              " Suggest to use torch.{0} or torch_npu.{0} instead."


def npu_format_cast_(self, format_or_tensor):
    warnings.warn(warning_str.format("npu_format_cast_"))
    return torch_npu.npu_format_cast_(self, format_or_tensor)


def npu_format_cast(self, format_or_tensor):
    warnings.warn(warning_str.format("npu_format_cast"))
    return torch_npu.npu_format_cast(self, format_or_tensor)


def npu_dtype_cast(self, dtype):
    warnings.warn(warning_str.format("npu_dtype_cast"))
    return torch_npu.npu_dtype_cast(self, dtype)


def npu_dtype_cast_(self, other):
    warnings.warn(warning_str.format("npu_dtype_cast_"))
    return torch_npu.npu_dtype_cast_(self, other)


def copy_memory_(self, src, non_blocking=False):
    warnings.warn(warning_str.format("copy_memory_"))
    return torch_npu.copy_memory_(self, src, non_blocking)


def one_(self):
    warnings.warn(warning_str.format("one_"))
    return torch_npu.one_(self)


def npu_confusion_transpose(self, perm, shape, transpose_first):
    warnings.warn(warning_str.format("npu_confusion_transpose"))
    return torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first)


@torch_device_guard
def _npu(self, *args, **kwargs):
    return torch_npu._C.npu(self, *args, **kwargs)


@property
def _is_npu(self):
    return torch_npu._C.is_npu(self)


def _type(self, *args, **kwargs):
    return torch_npu._C.type(self, *args, **kwargs)


@torch_device_guard
def _to(self, *args, **kwargs):
    return torch_npu._C.to(self, *args, **kwargs)


def _record_stream(self, *args, **kwargs):
    return torch_npu._C.record_stream(self, *args, **kwargs)


class NpuStorage(object):

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


storage_impl = torch.Tensor.storage

def _storage(self):
    if torch_npu._C.is_npu(self):
        return NpuStorage(torch_npu.get_storage_size(self))

    return storage_impl(self)


@torch_device_guard
def _new_empty(self, *args, **kwargs):
    if isinstance(args[0], int):
        list_args = list(args)
        sizes = []
        for item in list_args:
            if not isinstance(item, int):
                break
            sizes.append(item)
        args = tuple([tuple(sizes)] + list_args[len(sizes):])
    return torch_npu._C.new_empty(self, *args, **kwargs)


@torch_device_guard
def _new_empty_strided(self, *args, **kwargs):
    return torch_npu._C.new_empty_strided(self, *args, **kwargs)


@property
def _device(self):
    if torch_npu._C.is_npu(self):      
        return device(type='npu', index=self.get_device())
    return torch.device("cpu")


@torch_device_guard
def _new_full(self, *args, **kwargs):
    return torch_npu._C.new_full(self, *args, **kwargs)


@torch_device_guard
def _new_ones(self, *args, **kwargs):
    dst_device = kwargs.get("device", None)
    if dst_device is not None and "npu" in str(dst_device):
        kwargs["device"] = None
        return torch._C._TensorBase.new_ones(self, *args, **kwargs).to(dst_device)

    return torch._C._TensorBase.new_ones(self, *args, **kwargs)


@torch_device_guard
def _new_tensor(self, *args, **kwargs):
    if kwargs and "device" in kwargs:
        dst_device = kwargs.get("device")
        if "npu" in str(dst_device):
            args_requires_grad = kwargs.get("requires_grad", False)
            dtype = kwargs.get("dtype", self.dtype)
            if isinstance(args[0], torch.Tensor):
                res_tensor = args[0].clone().to(dtype=dtype, device=dst_device)
            elif isinstance(args[0], numpy.ndarray):
                res_tensor = torch.from_numpy(args[0]).to(dtype=dtype, device=dst_device)
            else:
                res_tensor = torch.from_numpy(numpy.array(args[0])).to(dtype=dtype, device=dst_device)

            if args_requires_grad:
                return res_tensor.detach().requires_grad_(args_requires_grad)
            else:
                return res_tensor.detach()

    return torch._C._TensorBase.new_tensor(self, *args, **kwargs)


@torch_device_guard
def _new_zeros(self, *args, **kwargs):
    if isinstance(args[0], int):
        list_args = list(args)
        sizes = []
        for item in list_args:
            if not isinstance(item, int):
                break
            sizes.append(item)
        args = tuple([tuple(sizes)] + list_args[len(sizes):])
    return torch_npu._C.new_zeros(self, *args, **kwargs)


def add_tensor_methods():
    torch.Tensor.npu_format_cast_ = npu_format_cast_
    torch.Tensor.npu_format_cast = npu_format_cast
    torch.Tensor.npu_dtype_cast = npu_dtype_cast
    torch.Tensor.npu_dtype_cast_ = npu_dtype_cast_
    torch.Tensor.copy_memory_ = copy_memory_
    torch.Tensor.one_ = one_
    torch.Tensor.npu_confusion_transpose = npu_confusion_transpose
    torch.Tensor.npu = _npu
    torch.Tensor.type = _type
    torch.Tensor.to = _to
    torch.Tensor.is_npu = _is_npu
    torch.Tensor.record_stream = _record_stream
    torch.Tensor.storage = _storage
    torch.Tensor.new_empty = _new_empty
    torch.Tensor.new_empty_strided = _new_empty_strided
    torch.Tensor.device = _device
    torch.Tensor.new_full = _new_full
    torch.Tensor.new_ones = _new_ones
    torch.Tensor.new_tensor = _new_tensor
    torch.Tensor.new_zeros = _new_zeros
    torch.Tensor.__reduce_ex__ = _reduce_ex
