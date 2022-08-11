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

from typing import Any
import torch
import torch._C as _C
import torch_npu

def _rebuild_npu_tensor(storage, npu_format, storage_offset, size, stride):
    t = torch.tensor([0], dtype=storage.dtype).to(storage.device)
    return t.npu_set_(storage, storage_offset, npu_format, size, stride)

def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, npu_format=2):
    if storage.device.type == 'npu':
        tensor = _rebuild_npu_tensor(storage, npu_format, storage_offset, size, stride)
    else:
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    return tensor

class _StorageBase(torch.storage._StorageBase):
    _cdata: Any
    is_cuda: bool = False
    is_npu = False
    is_sparse: bool = False

    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        """
        from torch.multiprocessing import get_sharing_strategy
        if self.is_cuda:
            pass  # CUDA doesn't use POSIX shared memory
        elif self.is_npu:
            pass
        elif get_sharing_strategy() == 'file_system':
            self._share_filename_()
        else:
            self._share_fd_()
        return self

def add_storage_methods():
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2