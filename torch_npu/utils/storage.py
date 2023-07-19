from typing import Any, Dict
from collections import OrderedDict
import torch
from torch.overrides import has_torch_function_unary, handle_torch_function, has_torch_function
from torch._namedtensor_internals import check_serializing_named_tensor

import torch_npu


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type != 'cpu':
        fake_tensor = torch.tensor([], device=self.device.type).set_(self)
        return fake_tensor.cpu().untyped_storage()
    else: 
        return self


def _deepcopy(self, memo):
    tmp_tensor = torch.tensor([], dtype=self.dtype, device=self._untyped_storage.device).set_(self)
    return tmp_tensor._typed_storage()

def add_storage_methods():
    torch.storage.UntypedStorage.cpu = _cpu
    torch.storage.TypedStorage._deepcopy = _deepcopy
