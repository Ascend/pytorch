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
  if (StorageDescHelper::CheckDescInit(src)) {
    StorageDescHelper::CopyDesc(self, src);
    return self;
  }
  // NPUStorageImpl create by constructor, NPUStorageDesc is not initialized by SetDesc.
  StorageDescHelper::SetDesc(self, self.unsafeGetTensorImpl()->sizes(),
                             self.unsafeGetTensorImpl()->strides());
  return self;
}

bool CheckStorageDesc(const at::Tensor& self, const c10::Storage src) {
  if (self.unsafeGetTensorImpl()->storage_offset() != 0) {
    return false;
  }
  if (!self.is_contiguous()) {
    return false;
  }
  int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
  int64_t nelements = c10::multiply_integers(self.unsafeGetTensorImpl()->sizes());
  if (new_size != nelements) {
    return false;
  }
  return true;
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
  if (StorageDescHelper::CheckDescInit(src)) {
    StorageDescHelper::CopyDesc(self, src);
    return self;
  }
  // NPUStorageImpl create by constructor, NPUStorageDesc is not initialized by SetDesc.
  if (CheckStorageDesc(self, src)) {
    StorageDescHelper::SetDesc(self, size, stride);
  } else {
    // Check input tensor propertys. If conditions are not met, NPUStorageDesc base_sizes_ change to 1D.
    // Conditions:
    // 1. Tensor storage_offset == 0
    // 2. Tnput tensor is contiguous
    // 3. Storage element size same to Tensor
    int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
    StorageDescHelper::SetDesc(self, {new_size}, {1});
  }
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
  c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl = c10::make_intrusive<torch_npu::NPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          0,
          c10_npu::NPUCachingAllocator::get()->allocate(0),
          c10_npu::NPUCachingAllocator::get(),
          true);
  c10::Storage storage(npu_storage_impl);
  set_storage_npu_(self, storage, 0, {0}, {});
  StorageDescHelper::SetDesc(self);
  TORCH_INTERNAL_ASSERT(dtype == self.dtype());
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
