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

#pragma once

#include <ATen/ATen.h>
#include <ATen/native/npu/utils/NpuUtils.h>
#include <TH/THTensor.hpp>
#include <c10/npu/NPUStream.h>
#include "ATen/native/npu/frame/StorageDescHelper.h"

namespace at {
namespace native {

static void storage_resize_npu(
    StorageImpl& storage,
    ptrdiff_t size,
    IntArrayRef new_size) {
  // AT_CHECK(size > 0, "invalid size");
  // AT_CHECK(storage.allocator() != nullptr, "Try to resize a tensor with
  // null allocator");
  if (!storage.resizable()) {
    AT_ERROR("Trying to resize storage that is not resizable");
    return;
  }

  at::DataPtr new_data;
  if (size != 0) {
    new_data = storage.allocator()->allocate(size);
  }
  at::DataPtr old_data = storage.set_data_ptr(std::move(new_data));
  ptrdiff_t old_size = storage.nbytes();
  storage.set_nbytes(size);
  npu::StorageDescHelper::UpdateDesc(storage.npu_desc_, new_size);

  if (old_data != nullptr) {
    ptrdiff_t copy_size = old_size;
    if (storage.nbytes() < copy_size) {
      copy_size = storage.nbytes();
    }
    if (copy_size > 0) {
      c10::npu::NPUStream copy_stream = c10::npu::getCurrentNPUStream();
      aclError error = aclrtMemcpyAsync(
          storage.data(),
          copy_size,
          old_data.get(),
          copy_size,
          ACL_MEMCPY_DEVICE_TO_DEVICE,
          copy_stream);
      if (error != ACL_ERROR_NONE) {
        AT_ERROR("ACL_Memcpy device to device error.");
        return;
      }
    }
  }
}

static inline void maybe_resize_storage_npu(
    TensorImpl* self,
    int64_t new_size,
    IntArrayRef size) {
  if (new_size > 0) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("Try to resize a tensor with null storage");
    }
    int64_t new_size_bytes = 
        (new_size + self->storage_offset()) * self->dtype().itemsize();
    if (new_size_bytes > self->storage().nbytes()) {
      storage_resize_npu(
          *(THTensor_getStoragePtr(self)),
          new_size_bytes,
          size);
    }
  }
}

inline TensorImpl* resize_impl_npu_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    for (size_t dim = 0; dim < size.size(); ++dim) {
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_npu(self, storage_size, size);

  return self;
}

static void resize_nd_npu(
    TensorImpl* self,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  // AT_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride != nullptr) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  resize_impl_npu_(self, sizes, strides);
}
} // namespace native
} // namespace at
