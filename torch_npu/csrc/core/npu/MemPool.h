#pragma once

#include <atomic>
#include <memory>

#include <c10/core/Allocator.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
using c10::CaptureId_t;
using c10::MempoolId_t;

// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the NPUCachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as a user-registered NPUPluggableAllocator.
struct C10_NPU_API MemPool {
    MemPool(
        std::shared_ptr<NPUCachingAllocator::NPUAllocator> allocator = nullptr,
        bool is_user_created = true);
    MemPool(const MemPool&) = delete;
    MemPool(MemPool&&) = default;
    MemPool& operator=(const MemPool&) = delete;
    MemPool& operator=(MemPool&&) = default;
    ~MemPool();

    MempoolId_t id();
    int use_count();
    c10::DeviceIndex device();
    static MempoolId_t graph_pool_handle(bool is_user_created = true);

private:
    static std::atomic<CaptureId_t> uid_;
    static std::atomic<CaptureId_t> uuid_;
    bool is_user_created_;
    MempoolId_t id_;
    c10::DeviceIndex device_;
};

} // namespace c10_npu
