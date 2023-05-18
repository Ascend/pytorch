from typing import Any, Dict
from collections import OrderedDict
import torch
from torch.overrides import has_torch_function_unary, handle_torch_function, has_torch_function
from torch._namedtensor_internals import check_serializing_named_tensor

import torch_npu


def _rebuild_npu_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, npu_storage_info):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    if tensor.is_cpu:
        return tensor
    
    torch_npu._C._npu_storage_set_desc(tensor, size, stride)
    if isinstance(npu_storage_info, bool):
        return tensor    
    tensor = torch_npu.npu_format_cast(tensor, npu_storage_info)
    return tensor


def _rebuild_tensor(storage, storage_offset, size, stride):
    tensor = torch.tensor([], dtype=storage.dtype, device=storage.device)
    tensor.set_(storage, storage_offset, size, stride)
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
            npu_origin_format = torch_npu._C._get_npu_origin_format(self)
            tmp_tensor = torch_npu.npu_format_cast(self, npu_origin_format).contiguous()
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
