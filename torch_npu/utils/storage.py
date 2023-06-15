from typing import Any, Dict
from collections import OrderedDict
import torch
from torch.overrides import has_torch_function_unary, handle_torch_function, has_torch_function
from torch._namedtensor_internals import check_serializing_named_tensor

import torch_npu


'''
Add patch to skip some unsupported method instead of reporting error
'''
def npu_share_memory_(self):
    """Moves the storage to shared memory.

    This is a no-op for storages already in shared memory and for CUDA
    storages, which do not need to be moved for sharing across processes.
    Storages in shared memory cannot be resized.

    Note that to mitigate issues like https://github.com/pytorch/pytorch/issues/95606
    it is thread safe to call this function from multiple threads on the same object.
    It is NOT thread safe though to call any other function on self without proper
    synchronization. Please see :doc:`/notes/multiprocessing` for more details.

    Returns: self
    """
    if self.is_npu:
        # While privateuse1 device is npu, share_memory is unsupported
        return self

    from torch.multiprocessing import get_sharing_strategy
    if self.is_cuda:
        pass  # CUDA doesn't use POSIX shared memory
    elif get_sharing_strategy() == 'file_system':
        self._share_filename_cpu_()
    else:
        self._share_fd_cpu_()
    return self


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type != 'cpu':
        fake_tensor = torch.tensor([], device=self.device.type).set_(self)
        return fake_tensor.cpu().untyped_storage()
    else: 
        return self


def add_storage_methods():
    torch.storage.UntypedStorage.cpu = _cpu
    torch.storage.UntypedStorage.share_memory_ = npu_share_memory_
