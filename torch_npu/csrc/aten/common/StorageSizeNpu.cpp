// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace at_npu {
namespace native {

  int64_t NPUNativeFunctions::get_storage_size(const at::Tensor& self) {
    torch_npu::utils::torch_check_npu(self);
    auto sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
    int64_t n = 1;
    for (auto s : sizes) {
        n *= s;
    }
    return n;
  }

  static void _npu_storage_resize_only(torch_npu::NPUStorageImpl& storage, ptrdiff_t size)
  {
    if (!storage.resizable()) {
    AT_ERROR("Trying to resize storage that is not resizable");
    return;
    }
    auto storage_desc = torch_npu::NPUBridge::GetNpuStorageImpl(&storage)->npu_desc_;
    size_t itemsize = storage_desc.data_type_.itemsize();

    at::DataPtr new_data = storage.allocator()->allocate(size);
    at::DataPtr old_data = storage.set_data_ptr(std::move(new_data));
    ptrdiff_t old_size = (ptrdiff_t)storage.nbytes();
    storage.set_nbytes(size);

    if (itemsize == 0) {
      AT_ERROR("When resizing, item size of storage cannot be zero.");
      return;
    }
    if ((size % (ptrdiff_t)itemsize) != 0) {
      AT_ERROR("The storage nbytes cannot be divided by item size, please check the specified size.");
      return;
    }
    std::vector<int64_t> resize_shape = {size/(ptrdiff_t)itemsize};
    // It is necessary to properly refresh the storage according to sizes and strides,
    // not just new sizes.
    at_npu::native::StorageDescHelper::UpdateDesc(
        torch_npu::NPUBridge::GetNpuStorageImpl(&storage)->npu_desc_, resize_shape, resize_shape);
    
    if (old_data != nullptr) {
      ptrdiff_t copy_size = old_size;
      if ((ptrdiff_t)storage.nbytes() < copy_size) {
        copy_size = (ptrdiff_t)storage.nbytes();
      }
      if (copy_size > 0) {
        aclError error = at_npu::native::CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
            storage,
            copy_size,
            old_data.get(),
            copy_size,
            ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (error != ACL_ERROR_NONE) {
          AT_ERROR("ACL_Memcpy device to device error.");
          return;
        }
      }
    }
  }

  static void _maybe_npu_storage_resize(at::TensorImpl* self, ptrdiff_t size)
  {
    if (!self->storage().unsafeGetStorageImpl()) {
      AT_ERROR("Try to resize a tensor with null storage");
      return;
    }
    _npu_storage_resize_only(*torch_npu::NPUBridge::GetNpuStorageImpl(self->storage().unsafeGetStorageImpl()), size);
  }

  at::Tensor NPUNativeFunctions::_npu_storage_resize(const at::Tensor& self, int64_t size) {
    int64_t new_size_bytes = (size + self.storage_offset()) * static_cast<int64_t>(self.dtype().itemsize());
    auto* self_impl = self.unsafeGetTensorImpl();
    _maybe_npu_storage_resize(self_impl, new_size_bytes);
    return self;
  }

} // namespace native
} // namespace at_npu