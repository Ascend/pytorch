#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/utils/LazyInit.h"
#endif

namespace torch_npu {

NPUStorageImpl::NPUStorageImpl(
    use_byte_size_t use_byte_size,
    size_t size_bytes,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable) : c10::StorageImpl(
      use_byte_size,
      size_bytes,
      at::DataPtr(std::move(data_ptr)),
      allocator,
      resizable)
{
}

void NPUStorageImpl::release_resources()
{
    StorageImpl::release_resources();
}

c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl_inner(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable)
{
    c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl = c10::make_intrusive<NPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size_bytes.as_int_unchecked(),
          std::move(data_ptr),
          allocator,
          resizable);
    return npu_storage_impl;
}

c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::Allocator* allocator,
    bool resizable)
{
#ifndef BUILD_LIBTORCH
    torch_npu::utils::npu_lazy_init();
#endif
  // Correctly create NPUStorageImpl object.
  c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl = make_npu_storage_impl_inner(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      std::move(allocator->allocate(size_bytes.as_int_unchecked())),
      allocator,
      resizable);
  // There is no need to consider the NPUStorageDesc information, it will be carried out in the subsequent processing.
  return npu_storage_impl;
}

}
