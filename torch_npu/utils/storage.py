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

from typing import Any, Dict, Tuple, Union
from collections import OrderedDict
import torch 
import torch._C as _C
from torch import _storage_classes
from torch.cuda import _CudaBase
from torch.overrides import has_torch_function_unary, handle_torch_function, has_torch_function
from torch._namedtensor_internals import check_serializing_named_tensor

import torch_npu
from . import serialization


def _rebuild_npu_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, npu_storage_info):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    if not serialization.RE_MAP_CPU:
        if isinstance(npu_storage_info, bool):
            tensor = tensor.npu()
        else:
            tensor = torch_npu.npu_format_cast(tensor.npu(), npu_storage_info)
    return tensor


def normalize_storage_type(storage):
    if isinstance(storage, torch.storage.TypedStorage):
        npu_flag = storage._untyped_storage.is_npu
    else:
        npu_flag = storage.is_npu
    return npu_flag


def _rebuild_tensor(storage, storage_offset, size, stride):
    tensor = torch.tensor([], dtype=storage.dtype, device=storage.device)
    tensor.set_(storage, storage_offset, size, stride)
    npu_flag = normalize_storage_type(storage)
    if npu_flag:
        tensor = tensor.npu()
    return tensor


def _reduce_ex(self, proto):
    if type(self) is torch.Tensor:
        if has_torch_function_unary(self):
            return handle_torch_function(torch.Tensor.__reduce_ex__, (self,), self, proto)
        check_serializing_named_tensor(self)
        torch.utils.hooks.warn_if_has_hooks(self)
        backward_hooks: Dict[Any, Any] = OrderedDict()
        if self.device.type == 'npu':
            npu_storage_format = torch_npu.get_npu_format(self)
            tmp_tensor = self.cpu()
            arg_npu  = (tmp_tensor.storage(),
                        tmp_tensor.storage_offset(),
                        tuple(tmp_tensor.size()),
                        tmp_tensor.stride(),
                        tmp_tensor.requires_grad,
                        backward_hooks,
                        npu_storage_format)
            return _rebuild_npu_tensor, arg_npu
        return self._reduce_ex_internal(proto)
    relevant_args = (self,)
    if type(self) is not torch.Tensor and has_torch_function(relevant_args):
        return handle_torch_function(torch.Tensor.__reduce_ex__, relevant_args, self, proto)
    func, args = self._reduce_ex_internal(proto)
    return torch._rebuild_from_type, (func, type(self), args, self.__dict__)


def add_storage_methods():
    torch._utils._rebuild_tensor = _rebuild_tensor
