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
import torch
import torch_npu

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


def npu_dtype_cast_(self, tensor):
    warnings.warn(warning_str.format("npu_dtype_cast_"))
    return torch_npu.npu_dtype_cast_(self, tensor)


def copy_memory_(self, src, non_blocking=False):
    warnings.warn(warning_str.format("copy_memory_"))
    return torch_npu.copy_memory_(self, src, non_blocking)


def one_(self):
    warnings.warn(warning_str.format("one_"))
    return torch_npu.one_(self)


def npu_confusion_transpose(self, perm, shape, transpose_first):
    warnings.warn(warning_str.format("npu_confusion_transpose"))
    return torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first)


def _npu(self, *args, **kwargs):
    warnings.warn(warning_str.format("npu"))
    return torch_npu._C.npu(self, *args, **kwargs)


def _type(self, *args, **kwargs):
    warnings.warn(warning_str.format("type"))
    return torch_npu._C.type(self, *args, **kwargs)


def _to(self, *args, **kwargs):
    warnings.warn(warning_str.format("to"))
    return torch_npu._C.to(self, *args, **kwargs)


def _is_npu(self):
    warnings.warn(warning_str.format("is_npu"))
    return torch_npu._C.is_npu(self)

def _record_stream(self, *args, **kwargs):
    warnings.warn(warning_str.format("_record_stream"))
    return torch_npu._C.record_stream(self, *args, **kwargs)

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