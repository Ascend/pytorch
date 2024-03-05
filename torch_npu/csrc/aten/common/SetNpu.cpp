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
#include <ATen/native/Resize.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
namespace at_npu {
namespace native {

torch_npu::NPUStorageImpl* storage_new_npu(caffe2::TypeMeta data_type) {
  c10::Allocator *allocator = c10_npu::NPUCachingAllocator::get();
  torch_npu::NPUStorageImpl* storage =
      c10::make_intrusive<torch_npu::NPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          0,
          allocator->allocate(0),
          allocator,
          true)
          .release();
  return storage;
}

void set_storage_nd_npu(
    at::Tensor& self,
    c10::Storage storage,
    int64_t storage_offset,
    int nDimension,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  at::native::checkSetStorage(self, storage, storage_offset, size, stride);

  if (storage_offset < 0) {
    AT_ERROR("at::Tensor: invalid storage offset");
  }
  self.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  resize_nd_npu(self.unsafeGetTensorImpl(), nDimension, size.data(), stride.data());
}

void set_storage_npu_(
    at::Tensor& self,
    c10::Storage storage_,
    long storageOffset_,
    c10::IntArrayRef size_,
    c10::IntArrayRef stride_) {
  if (stride_.data()) {
  }
  set_storage_nd_npu(
      self,
      storage_,
      storageOffset_,
      size_.size(),
      size_,
      stride_);
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self, c10::Storage src) {
  int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
  set_storage_npu_(
      self,
      src,
      0,
      {new_size},
      {});
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

at::Tensor& NPUNativeFunctions::set_(
    at::Tensor& self,
    c10::Storage src,
    long storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  set_storage_npu_(
      self,
      src,
      storage_offset,
      size,
      stride);
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

at::Tensor& set_format_npu_(
    at::Tensor& self,
    c10::Storage src,
    long storage_offset,
    long npu_format,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  set_storage_npu_(
      self,
      src,
      storage_offset,
      size,
      stride);

  StorageDescHelper::SetDesc(self, size, stride, (aclFormat)npu_format);
  return self;
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self) {
  caffe2::TypeMeta dtype = self.dtype();
  c10::Storage storage(
      c10::Storage::use_byte_size_t(),
      0,
      c10_npu::NPUCachingAllocator::get(),
      true);

  set_storage_npu_(self, storage, 0, {0}, {});
  StorageDescHelper::SetDesc(self);
  TORCH_INTERNAL_ASSERT(dtype == self.dtype(), OPS_ERROR(ErrCode::TYPE));
  return self;
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self, const at::Tensor& src) {
  at::TensorImpl* self_ = self.unsafeGetTensorImpl();
  at::TensorImpl* src_ = src.unsafeGetTensorImpl();
  if (self_ != src_) {
    set_storage_nd_npu(
        self,
        src.storage(),
        src.storage_offset(),
        src.dim(),
        src.sizes(),
        src.strides());
  }
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

} // namespace native
} // namespace at_npu
