#include <mutex>
#include <unordered_map>
#include <utility>

#include "torch_npu/csrc/npu/NPUPluggableAllocator.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace torch::npu::NPUPluggableAllocator {

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata()
    : size(0), device_idx(-1), stream{} {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    int device_idx,
    aclrtStream stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
NPUPluggableAllocator::NPUPluggableAllocator(
    std::function<void*(size_t, int, aclrtStream)> alloc_fn,
    std::function<void(void*, size_t, int, aclrtStream)> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

NPUPluggableAllocator::NPUPluggableAllocator(NPUPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      erase_stream_fn_(other.erase_stream_fn_) {}

void NPUPluggableAllocator::set_init_fn(std::function<void(int)> init_fn)
{
    init_fn_ = std::move(init_fn);
}

void NPUPluggableAllocator::set_reset_fn(std::function<void(bool)> reset_fn)
{
    reset_fn_ = std::move(reset_fn);
}

void NPUPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn)
{
    memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void NPUPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn)
{
    base_alloc_fn_ = std::move(base_alloc_fn);
}

void NPUPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, c10_npu::NPUStream stream)> record_stream_fn)
{
    record_stream_fn_ = std::move(record_stream_fn);
}

void NPUPluggableAllocator::set_erase_stream_fn(
    std::function<void(void* ptr, c10_npu::NPUStream stream)> erase_stream_fn)
{
    erase_stream_fn_ = std::move(erase_stream_fn);
}

void NPUPluggableAllocator::set_get_device_stats_fn(
    std::function<c10_npu::NPUCachingAllocator::DeviceStats(int)> get_device_stats_fn)
{
    get_device_stats_fn_ = std::move(get_device_stats_fn);
}

void NPUPluggableAllocator::set_reset_peak_status_fn(
    std::function<void(int)> reset_peak_status_fn)
{
    reset_peak_status_fn_ = std::move(reset_peak_status_fn);
}

void* NPUPluggableAllocator::malloc(
    size_t size,
    int device,
    aclrtStream stream)
{
    void* r = alloc_fn_(size, device, stream);
    {
        const std::lock_guard<std::mutex> lock(allocator_mutex_);
        allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
    }
    return r;
}

c10::DataPtr NPUPluggableAllocator::allocate(size_t size)
{
    int device = -1;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    aclrtStream stream = c10_npu::getCurrentNPUStreamNoWait(device);
    void* r =
        this->malloc(size, device, stream);
    c10::DataPtr data_ptr = {
        r,
        r,
        raw_deleter(),
        c10::Device(
            c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device))};
    return data_ptr;
}

c10::DataPtr NPUPluggableAllocator::allocate_with_aligned(size_t size, size_t base_addr_aligned_kb) const
{
    TORCH_CHECK(false, "NPUPluggableAllocator does't has allocate_with_aligned", PTA_ERROR(ErrCode::NOT_SUPPORT));
    return c10::DataPtr();
}

c10::DeleterFnPtr NPUPluggableAllocator::raw_deleter() const
{
    return &custom_raw_deleter;
}

void* NPUPluggableAllocator::raw_alloc(size_t nbytes)
{
    int device = -1;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    aclrtStream stream = c10_npu::getCurrentNPUStreamNoWait(device);
    return malloc(nbytes, device, stream);
}

void* NPUPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    aclrtStream stream)
{
    int device = -1;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    return malloc(nbytes, device, stream);
}

void NPUPluggableAllocator::raw_delete(void* ptr)
{
    aclrtStream stream{};
    int device_idx = -1;
    size_t size = 0;
    {
        const std::lock_guard<std::mutex> lock(allocator_mutex_);
        TORCH_CHECK(
            allocation_metadata_.count(ptr),
            "Trying to free a pointer not allocated here", PTA_ERROR(ErrCode::PTR));
        _AllocationMetadata& metadata = allocation_metadata_[ptr];
        size = metadata.size;
        device_idx = metadata.device_idx;
        stream = metadata.stream;
        allocation_metadata_.erase(ptr);
    }
    free_fn_(ptr, size, device_idx, stream);
}

void NPUPluggableAllocator::init(int device_count)
{
    if (init_fn_) {
        init_fn_(device_count);
    }
    initialized_ = true;
}

bool NPUPluggableAllocator::initialized()
{
    return initialized_;
}

void NPUPluggableAllocator::setMemoryFraction(double fraction, int device)
{
    if (memory_fraction_fn_) {
        memory_fraction_fn_(fraction, device);
    }
}

void NPUPluggableAllocator::emptyCache(bool check_error)
{
    if (reset_fn_) {
        return reset_fn_(check_error);
    }
}

void NPUPluggableAllocator::cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock)
{
    TORCH_NPU_WARN("NPUPluggableAllocator does not yet support cacheInfo. "
                  "If you need it, please file an issue describing your use case.");
}

void* NPUPluggableAllocator::getBaseAllocation(void* ptr, size_t* size)
{
    if (base_alloc_fn_) {
        return base_alloc_fn_(ptr, size);
    } else {
        return ptr;
    }
}

void NPUPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    streamType stream)
{
    if (record_stream_fn_) {
        record_stream_fn_(ptr.get(), stream);
    }
}

void NPUPluggableAllocator::eraseStream(
    const c10::DataPtr& ptr,
    streamType stream)
{
    if (erase_stream_fn_) {
        erase_stream_fn_(ptr.get(), stream);
    }
}

c10_npu::NPUCachingAllocator::DeviceStats NPUPluggableAllocator::getDeviceStats(int device)
{
    if (get_device_stats_fn_) {
        return get_device_stats_fn_(device);
    } else {
        TORCH_NPU_WARN("get_device_stats_fn_ is not define, please set by set_get_device_stats_fn");
    }
}

void NPUPluggableAllocator::resetAccumulatedStats(int device)
{
    TORCH_NPU_WARN("NPUPluggableAllocator does not yet support resetAccumulatedStats. "
                  "If you need it, please file an issue describing your use case.");
}

void NPUPluggableAllocator::resetPeakStats(int device)
{
    if (reset_peak_status_fn_) {
        reset_peak_status_fn_(device);
    } else {
        TORCH_NPU_WARN("reset_peak_status_fn_ is not define, please set by set_reset_peak_status_fn");
    }
}

c10_npu::NPUCachingAllocator::SnapshotInfo NPUPluggableAllocator::snapshot()
{
    TORCH_NPU_WARN("NPUPluggableAllocator does not yet support snapshot. "
                  "If you need it, please file an issue describing your use case.");
}

// CUDAGraph interactions
void NPUPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10_npu::MempoolId_t mempool_id,
    std::function<bool(aclrtStream)> filter)
{
    TORCH_CHECK(
        false,
        "NPUPluggableAllocator does not yet support beginAllocateToPool. "
        "If you need it, please file an issue describing your use case.");
}

void NPUPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10_npu::MempoolId_t mempool_id)
{
    TORCH_CHECK(
        false,
        "NPUPluggableAllocator does not yet support endAllocateToPool. "
        "If you need it, please file an issue describing your use case.");
}

void NPUPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    c10_npu::MempoolId_t mempool_id)
{
    TORCH_CHECK(
        false,
        "NPUPluggableAllocator does not yet support releasePool. "
        "If you need it, please file an issue describing your use case.");
}

void NPUPluggableAllocator::FreeDeviceCachedMemory(int device)
{
    TORCH_NPU_WARN("NPUPluggableAllocator does not yet support FreeDeviceCachedMemory. "
                  "If you need it, please file an issue describing your use case.");
}

std::string NPUPluggableAllocator::name()
{
    return "pluggable";
}

// Note [COW/lazy_clone is not supported yet]
void NPUPluggableAllocator::copy_data(void* dest, const void* src, std::size_t count) const
{
    default_copy_data(dest, src, count);
}
void NPUPluggableAllocator::recordHistory(
    bool enabled,
    c10_npu::NPUCachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10_npu::NPUCachingAllocator::RecordContext when)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
}

void NPUPluggableAllocator::attachOutOfMemoryObserver(
    c10_npu::NPUCachingAllocator::OutOfMemoryObserver observer)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support attachOutOfMemoryObserver. "
        "If you need it, please file an issue describing your use case.");
}

bool NPUPluggableAllocator::checkUceInMemPool(int device)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support checkUceInMemPool. "
        "If you need it, please file an issue describing your use case.");
    return false;
}

bool NPUPluggableAllocator::checkBlockIsSafe(const c10::DataPtr& ptr)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support checkBlockIsSafe. "
        "If you need it, please file an issue describing your use case.");
    return false;
}

void NPUPluggableAllocator::markAllBlockUnsafe(int device)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support markAllBlockUnsafe. "
        "If you need it, please file an issue describing your use case.");
}

void NPUPluggableAllocator::updateBlockToSafe(const c10::DataPtr& ptr)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support updateBlockToSafe. "
        "If you need it, please file an issue describing your use case.");
}

void NPUPluggableAllocator::cleanEvent()
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support cleanEvent. "
        "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<c10_npu::NPUCachingAllocator::AllocatorState> NPUPluggableAllocator::getCheckpointState(c10::DeviceIndex device, c10_npu::MempoolId_t id)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support getCheckpointState. "
        "If you need it, please file an issue describing your use case.");
}

c10_npu::NPUCachingAllocator::CheckpointDelta NPUPluggableAllocator::setCheckpointPoolState(c10::DeviceIndex device, std::shared_ptr<c10_npu::NPUCachingAllocator::AllocatorState> pps)
{
    TORCH_NPU_WARN(
        "NPUPluggableAllocator does not yet support setCheckpointPoolState. "
        "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator>
    current_custom_allocator;

std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator> getCurrentAllocator()
{
    return current_custom_allocator;
}

std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator> createCustomAllocator(
    std::function<void*(size_t, int, aclrtStream)> alloc_fn,
    std::function<void(void*, size_t, int, aclrtStream)> free_fn)
{
    std::shared_ptr<NPUPluggableAllocator> allocator(
        new NPUPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device_count));
    allocator->init(device_count);
    return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator>&
        allocator)
{
    TORCH_CHECK(
        !c10_npu::NPUCachingAllocator::allocator.load()->initialized(),
        "Can't swap an already initialized allocator", PTA_ERROR(ErrCode::PTR));
    c10_npu::NPUCachingAllocator::allocator.store(allocator.get());
    current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr)
{
    current_custom_allocator->raw_delete(ptr);
}

} // namespace torch::npu::NPUPluggableAllocator
