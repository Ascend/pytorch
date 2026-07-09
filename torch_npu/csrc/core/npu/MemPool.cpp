#include "torch_npu/csrc/core/npu/MemPool.h"

namespace c10_npu {

// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10_npu::MemPool().
//
// uuid_ is incremented when NPUGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or NPUGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> MemPool::uid_{ 1 };
std::atomic<CaptureId_t> MemPool::uuid_{ 1 };


MemPool::MemPool(
    std::shared_ptr<NPUCachingAllocator::NPUAllocator> allocator,
    bool is_user_created,
    bool use_on_oom,
    bool no_split)
    : is_user_created_(is_user_created)
{
    if (is_user_created_) {
        id_ = {0, uid_++};
    } else {
        id_ = {uuid_++, 0};
    }
    device_ = c10_npu::current_device();
    NPUCachingAllocator::createOrIncrefPool(device_, id_, std::move(allocator));
    if (use_on_oom) {
        NPUCachingAllocator::setUseOnOOM(device_, id_, true);
    }
    if (no_split) {
        NPUCachingAllocator::setNoSplit(device_, id_);
    }
}

MemPool::~MemPool() {
    // TORCH_INTERNAL_ASSERT(use_count() == 1);
    // We used to assert that TORCH_INTERNAL_ASSERT(use_count() == 1);
    // However, this assertion is not true if a memory pool is shared
    // with a cuda graph. That CUDAGraph will increase the use count
    // until it is reset.
    NPUCachingAllocator::setUseOnOOM(device_, id_, false);
    NPUCachingAllocator::releasePool(device_, id_);
    NPUCachingAllocator::emptyCache(id_);
}

MempoolId_t MemPool::id()
{
    return id_;
}

int MemPool::use_count()
{
    return NPUCachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device()
{
    return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created)
{
    if (is_user_created) {
        return {0, uid_++};
    }
    return {uuid_++, 0};
}

} // namespace c10_npu
