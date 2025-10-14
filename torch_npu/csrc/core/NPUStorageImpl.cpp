#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

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
    std::lock_guard<std::mutex> lock(unique_id_mutex_);
    static uint64_t idx = 0;
    unique_id_ = idx++;
}

void NPUStorageImpl::release_resources()
{
    StorageImpl::release_resources();
}

c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable)
{
    int64_t size = size_bytes.as_int_unchecked();
    TORCH_CHECK(size >= 0, "Size bytes must be non-negative, but got ", size);

    if (size == 0) {
        TORCH_CHECK(data_ptr == nullptr, "When size is 0, data_ptr must be null.");
    } else if (data_ptr == nullptr) {
        TORCH_CHECK(allocator != nullptr, "When data_ptr is null and size > 0, allocator must be provided.");
        data_ptr = allocator->allocate(size);
        TORCH_CHECK(data_ptr, "Get data_ptr failed", PTA_ERROR(ErrCode::PARAM));
    }

    // Correctly create NPUStorageImpl object.
    c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl = c10::make_intrusive<NPUStorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes.as_int_unchecked(),
        std::move(data_ptr),
        allocator,
        resizable);
    // There is no need to consider the NPUStorageDesc information, it will be carried out in the subsequent processing.
    return npu_storage_impl;
}

} // namespace torch_npu
