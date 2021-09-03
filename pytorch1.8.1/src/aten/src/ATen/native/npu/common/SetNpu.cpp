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

// #pragma once
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/npu/common/ResizeNpu.h>
#include <TH/THTensor.hpp>
#include <c10/npu/NPUCachingAllocator.h>
#include "ATen/native/npu/frame/StorageDescHelper.h"
#include <torch/script.h>

namespace at {
namespace native {

using namespace at::native::npu;

StorageImpl* storage_new_npu(caffe2::TypeMeta data_type) {
  StorageImpl* storage =
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          0,
          at::npu::NPUCachingAllocator::get(),
          true)
          .release();
  return storage;
}

void set_storage_nd_npu(
    TensorImpl* self,
    StorageImpl* storage,
    ptrdiff_t storageOffset,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  if (THTensor_getStoragePtr(self) != storage) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    auto data_type = self->dtype();
    if (storage != nullptr) {
      c10::raw::intrusive_ptr::incref(storage);
      THTensor_stealAndSetStoragePtr(self, storage);
    } else {
      THTensor_stealAndSetStoragePtr(self, storage_new_npu(data_type));
    }
  }

  if (storageOffset < 0) {
    AT_ERROR("Tensor: invalid storage offset");
  }
  self->set_storage_offset(storageOffset);
  resize_nd_npu(self, nDimension, size, stride);
}

void set_storage_npu_(
    TensorImpl* self,
    StorageImpl* storage_,
    ptrdiff_t storageOffset_,
    at::IntArrayRef size_,
    at::IntArrayRef stride_) {
  if (stride_.data()) {
    // AT_CHECK(size_.size() == stride_.size(), "inconsistent size/stride
    // sizes");
  }
  set_storage_nd_npu(
      self,
      storage_,
      storageOffset_,
      size_.size(),
      size_.data(),
      stride_.data());
}

Tensor& set_source_Storage_npu_(Tensor& self, Storage src) {
  int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
  set_storage_npu_(
      self.unsafeGetTensorImpl(),
      src.unsafeGetStorageImpl(),
      0,
      {new_size},
      {});
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

Tensor& set_source_storage_offset_npu_(
    Tensor& self,
    Storage src,
    long storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  set_storage_npu_(
      self.unsafeGetTensorImpl(),
      src.unsafeGetStorageImpl(),
      storage_offset,
      size,
      stride);
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

Tensor& set_format_npu_(
    Tensor& self,
    Storage src,
    long storage_offset,
    long npu_format,
    IntArrayRef size,
    IntArrayRef stride) {
  set_storage_npu_(
      self.unsafeGetTensorImpl(),
      src.unsafeGetStorageImpl(),
      storage_offset,
      size,
      stride);

  StorageDescHelper::SetDesc(self, size, stride, (aclFormat)npu_format);
  return self;
}

Tensor& set_npu_(Tensor& self) {
  set_storage_npu_(self.unsafeGetTensorImpl(), NULL, 0, {0}, {});
  StorageDescHelper::SetDesc(self);
  return self;
}

Tensor& set_source_tensor_npu_(Tensor& self, const Tensor& src) {
  TensorImpl* self_ = self.unsafeGetTensorImpl();
  TensorImpl* src_ = src.unsafeGetTensorImpl();
  if (self_ != src_) {
    set_storage_nd_npu(
        self_,
        THTensor_getStoragePtr(src_),
        src_->storage_offset(),
        src_->dim(),
        THTensor_getSizePtr(src_),
        THTensor_getStridePtr(src_));
  }
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("set_.source_Storage", TORCH_FN(set_source_Storage_npu_));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(set_source_storage_offset_npu_));
  m.impl("set_.source_Tensor", TORCH_FN(set_source_tensor_npu_));
  m.impl("set_", TORCH_FN(set_npu_));
}

} // namespace native
} // namespace at
