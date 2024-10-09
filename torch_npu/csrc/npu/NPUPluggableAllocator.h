#pragma once

#include <c10/core/Allocator.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

#include <array>
#include <mutex>

namespace torch::npu::NPUPluggableAllocator {

using streamType = c10_npu::NPUStream;

std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator> getCurrentAllocator();
std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator> createCustomAllocator(
    std::function<void*(size_t, int, aclrtStream)> alloc_fn,
    std::function<void(void*, size_t, int, aclrtStream)> free_fn);
void changeCurrentAllocator(
    const std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator>&
        allocator);

struct _AllocationMetadata {
    _AllocationMetadata();
    _AllocationMetadata(size_t size, int device_idx, aclrtStream stream);
    size_t size;
    int device_idx;
    aclrtStream stream;
};

struct NPUPluggableAllocator
    : public c10_npu::NPUCachingAllocator::NPUAllocator {
    NPUPluggableAllocator(
        std::function<void*(size_t, int, aclrtStream)> alloc_fn,
        std::function<void(void*, size_t, int, aclrtStream)> free_fn);

    NPUPluggableAllocator(NPUPluggableAllocator& other);

    void set_init_fn(std::function<void(int)> init_fn);
    void set_reset_fn(std::function<void(bool)> reset_fn);
    void set_memory_fraction_fn(
        std::function<void(double, int)> memory_fraction_fn);
    void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);
    void set_record_stream_fn(
        std::function<void(void* ptr, aclrtStream stream)> record_stream_fn);
    void set_erase_stream_fn(
        std::function<void(void* ptr, aclrtStream stream)> erase_stream_fn);
    void set_get_device_stats_fn(std::function<c10_npu::NPUCachingAllocator::DeviceStats(int)> get_device_stats_fn);
    void set_reset_peak_status_fn(std::function<void(int)> reset_peak_status_fn);
    void set_snapshot_fn(std::function<c10_npu::NPUCachingAllocator::SnapshotInfo()> snapshot_fn);
    void* malloc(size_t size, int device, aclrtStream stream);

    c10::DataPtr allocate(size_t size) const override;
    c10::DeleterFnPtr raw_deleter() const override;

    void* raw_alloc(size_t nbytes) override;
    void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream) override;
    void raw_delete(void* ptr) override;
    void init(int device_count) override;
    bool initialized() override;
    void setMemoryFraction(double fraction, int device) override;
    void emptyCache(bool check_error) override;
    void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) override;
    void* getBaseAllocation(void* ptr, size_t* size) override;
    void recordStream(const c10::DataPtr&, streamType stream) override;
    void eraseStream(const c10::DataPtr&, streamType stream) override;
    c10_npu::NPUCachingAllocator::DeviceStats getDeviceStats(
        int device) override;
    void resetAccumulatedStats(int device) override;
    void resetPeakStats(int device) override;
    c10_npu::NPUCachingAllocator::SnapshotInfo snapshot() override;
    void FreeDeviceCachedMemory(int device) override;
    std::string name() override;
    void recordHistory(
        bool enabled,
        c10_npu::NPUCachingAllocator::CreateContextFn context_recorder,
        size_t alloc_trace_max_entries,
        c10_npu::NPUCachingAllocator::RecordContext when) override;
    void attachOutOfMemoryObserver(c10_npu::NPUCachingAllocator::OutOfMemoryObserver observer) override;
    bool checkUceInMemPool(int device) override;
    bool checkBlockIsSafe(const c10::DataPtr& ptr) override;
    void markAllBlockUnsafe(int device) override;
    void updateBlockToSafe(const c10::DataPtr &ptr) override;
    void cleanEvent() override;

protected:
    std::function<void*(size_t, int, aclrtStream)> alloc_fn_;
    std::function<void(void*, size_t, int, aclrtStream)> free_fn_;
    std::function<void(int)> init_fn_;
    std::function<void(bool)> reset_fn_;
    std::function<void(double, int)> memory_fraction_fn_;
    std::function<void*(void*, size_t*)> base_alloc_fn_;
    std::function<void(void* ptr, aclrtStream stream)> record_stream_fn_;
    std::function<void(void* ptr, aclrtStream stream)> erase_stream_fn_;
    std::function<c10_npu::NPUCachingAllocator::DeviceStats(int)> get_device_stats_fn_;
    std::function<void(int)> reset_peak_status_fn_;
    std::function<c10_npu::NPUCachingAllocator::SnapshotInfo()> snapshot_fn_;
    std::mutex allocator_mutex_;
    // We do the bookeeping here in order to simplify custom allocators
    std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

    bool initialized_ = false;
};
} // namespace torch::npu::NPUPluggableAllocator
