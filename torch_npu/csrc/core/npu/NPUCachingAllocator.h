#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"

#include <mutex>
#include <atomic>

inline std::shared_ptr<npu_logging::Logger>& GetLoggerMem()
{
    static std::shared_ptr<npu_logging::Logger> loggerMem = npu_logging::logging().getLogger("torch_npu.memory");
    return loggerMem;
}
// macros for log memory
#define TORCH_NPU_MEMORY_LOGD(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGD(GetLoggerMem(), format, ##__VA_ARGS__);                 \
        ASCEND_LOGD(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_MEMORY_LOGI(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGI(GetLoggerMem(), format, ##__VA_ARGS__);                 \
        ASCEND_LOGI(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_MEMORY_LOGW(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGW(GetLoggerMem(), format, ##__VA_ARGS__);                 \
        ASCEND_LOGW(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_MEMORY_LOGE(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGE(GetLoggerMem(), format, ##__VA_ARGS__);                 \
        ASCEND_LOGE(format, ##__VA_ARGS__);                                    \
    } while (0);

std::string format_size(uint64_t size);

namespace c10_npu {
constexpr size_t kAclIpcHandleSize = 65;
namespace NPUCachingAllocator {

C10_NPU_API std::mutex* getFreeMutex();

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class FreeMemoryCallback {
public:
    virtual ~FreeMemoryCallback(){};
    virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeNPUMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeNPUMemoryCallbacksRegistry, name, __VA_ARGS__);


// (Ascend): Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (NPUCachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THNCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.
using c10::CachingAllocator::Stat;
using c10::CachingAllocator::StatType;
using c10::CachingAllocator::StatTypes;
using c10::CachingAllocator::StatArray;
using c10::CachingAllocator::for_each_selected_stat_type;
using c10::CachingDeviceAllocator::DeviceStats;

typedef std::shared_ptr<c10::GatheredContext> (*CreateContextFn)(void);

// Struct containing info of an allocation block (i.e. a fractional part of a cudaMalloc)..
struct BlockInfo {
    int64_t size = 0;
    int64_t requested_size = 0;
    int32_t gc_counter = 0;
    bool allocated = false;
    bool active = false;
    std::shared_ptr<c10::GatheredContext> context_when_allocated;
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
    int64_t device = 0;
    int64_t  address = 0;
    aclrtStream stream = nullptr;
    int64_t total_size = 0;
    int64_t requested_size = 0;
    int64_t allocated_size = 0;
    int64_t active_size = 0;
    bool is_large = false;
    bool is_expandable = false;
    MempoolId_t owner_private_pool_id = {0, 0};
    std::vector<BlockInfo> blocks;
    std::shared_ptr<c10::GatheredContext> context_when_allocated;
};

struct AllocatorState {
    virtual ~AllocatorState() = default;
};

struct TraceEntry {
    enum Action {
        ALLOC,          // API made to the caching allocator for new memory
        FREE_REQUESTED, // API call made to the caching allocator to free memory
        FREE_COMPLETED, // The allocator might have to delay a free because
                        // it is still in use on another stream via
                        // record_stream This event is generated when a free
                        // actually completes.
        SEGMENT_ALLOC, // a call to AclrtMalloc to get more memory from the OS
        SEGMENT_FREE, // a call to aclrtFree to return memory to the OS (e.g. to
                      // defragment or empty_caches)
        SEGMENT_MAP,  // a call to AclrtMapMem (used with expandable_segments)
        SEGMENT_UNMAP, // unmap part of a segment (used with expandable
                       // segments)
        SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to
                  // trace events
        OOM, // the allocator threw an OutOfMemoryError (addr_ is the amount of
            // free bytes reported by cuda)
        WORKSPACE_SNAPSHOT
    };
    TraceEntry(Action action, int device, int64_t addr, size_t size,
               aclrtStream stream,
               MempoolId_t mempool = {0, 0},
               std::shared_ptr<c10::GatheredContext> context = nullptr)
        : action_(action), device_(device), addr_(addr),
          context_(std::move(context)), stream_(stream), size_(size),
          mempool_(std::move(mempool))
    {}
    Action action_;
    int device_;
    int64_t addr_; // for OOM, this is the amount of free bytes reported by cuda
    std::shared_ptr<c10::GatheredContext> context_;
    aclrtStream stream_;
    int64_t size_;
    MempoolId_t mempool_;
};

struct SnapshotInfo {
    std::vector<SegmentInfo> segments;
    std::vector<std::vector<TraceEntry> > device_traces;
};

// returns the pointers freed in the pool
// and the pointers allocated. Note: a pointer
// may appear in both freed and allocated
struct CheckpointDelta {
    std::vector<void*> ptrs_freed;
    std::vector<at::DataPtr> dataptrs_allocd;
};

enum struct RecordContext {
    NEVER = 0,
    STATE = 1, // only keep stacks for active allocations
    ALLOC = 2, // additionally keep stacks for allocations in the trace history
    ALL = 3,   // additionally record stacks for when something is freed
};

using OutOfMemoryObserver =
    std::function<void(int64_t device, int64_t allocated, int64_t device_total,
                       int64_t device_free)>;
// Observer called when an allocation is preemptively rejected due to
// throw_on_npumalloc_oom policy. Parameters:
//   - device: device index
//   - alloc_size: size of the rejected allocation request
//   - total_allocated: total memory allocated before the request
//   - device_total: total device memory
using OomRejectionObserver =
    std::function<void(int64_t device, size_t alloc_size, size_t total_allocated,
                       size_t device_total)>;
using AllocatorTraceTracker = std::function<void(const TraceEntry&)>;

struct ShareableHandle {
    ptrdiff_t offset;
    std::string handle;
};

class NPUAllocator : public c10::DeviceAllocator {
public:
    virtual c10::DataPtr allocate_with_aligned(size_t size, size_t aligned) const = 0;
    virtual void* raw_alloc(size_t nbytes) = 0;
    virtual void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream) = 0;
    virtual void raw_delete(void* ptr) = 0;
    virtual void init(int device_count) = 0;
    virtual double getMemoryFraction(int device) = 0;
    virtual void setMemoryFraction(double fraction, int device) = 0;
    virtual void emptyCacheImpl(bool check_error, bool free_physical) = 0;
    virtual void emptyCache(bool check_error) = 0;
    virtual void emptyVirtAddrCache(bool check_error) = 0;
    virtual void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) = 0;
    virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
    virtual void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) = 0;
    void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override {
        TORCH_CHECK(
            stream.device_type() == c10::DeviceType::PrivateUse1,
            "Expected NPU (PrivateUse1) stream, got ",
            c10::DeviceTypeName(stream.device_type()),
            PTA_ERROR(ErrCode::PARAM));
        NPUStream npu_stream = NPUStream(stream);
        recordStream(ptr, npu_stream);
    }
    virtual void eraseStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) = 0;
    virtual void eraseStreamWithBlockPtr(void* block_ptr, c10_npu::NPUStream stream, void* work_ptr) = 0;
    virtual void* getBlockPtr(const c10::DataPtr& ptr) = 0;
    virtual void recordHcclWorkForBlock(void* block_ptr, void* work_ptr) = 0;
    std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override {
        c10_npu::NPUGuard device_guard(device);
        size_t free = 0;
        size_t total = 0;
        NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
        return {free, total};
    }
    virtual SnapshotInfo snapshot() = 0;

    virtual void beginAllocateToPool(
        c10::DeviceIndex device,
        MempoolId_t mempool_id,
        std::function<bool(aclrtStream)> filter) = 0;
    virtual void endAllocateToPool(
        c10::DeviceIndex device,
        MempoolId_t mempool_id) = 0;
    virtual void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) = 0;
    virtual int getPoolUseCount(
        c10::DeviceIndex device,
        MempoolId_t mempool_id)
    {
        TORCH_CHECK(false, name(),
            " does not yet support getPoolUseCount. "
            "If you need it, please file an issue describing your use case.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    }

    virtual bool hasCapturesUnderway(c10::DeviceIndex device) { return false; }
    virtual void FreeDeviceCachedMemory(int device) = 0;
    virtual std::string name() = 0;
    virtual bool checkPoolLiveAllocations(
        c10::DeviceIndex device,
        MempoolId_t mempool_id,
        const std::unordered_set<void*>& expected_live_allocations)
    {
        TORCH_CHECK(false, name(),
            " does not yet support checkPoolLiveAllocations. "
            "If you need it, please file an issue describing your use case.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    virtual ShareableHandle shareIpcHandle(void* ptr) = 0;
    virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) = 0;
    virtual bool isHistoryEnabled()
    {
        TORCH_CHECK(
            false, name(),
            " does not yet support recordHistory. "
            "If you need it, please file an issue describing your use case.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    virtual void recordHistory(bool enabled, CreateContextFn context_recorder,
                               size_t alloc_trace_max_entries,
                               RecordContext when) = 0;
    virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;
    virtual void attachOomRejectionObserver(OomRejectionObserver observer) = 0;
    virtual bool checkUceInMemPool(int device) = 0;
    virtual bool checkBlockIsSafe(const c10::DataPtr& ptr) = 0;
    virtual void markAllBlockUnsafe(int device) = 0;
    virtual void updateBlockToSafe(const c10::DataPtr &ptr) = 0;
    virtual void cleanEvent() = 0;
    virtual void buildServerMemMapForHccl(int device, std::shared_ptr<c10d_npu::HCCLComm> hcclComm) {}
    virtual std::shared_ptr<AllocatorState> getCheckpointState(
        c10::DeviceIndex device,
        MempoolId_t id) = 0;
    virtual CheckpointDelta setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<AllocatorState> pps) = 0;
    // Attached AllocatorTraceTracker callbacks will be called while the
    // per-device allocator lock is held. Any additional locks taken from within
    // the callback must be proven to always have the lock order that never
    // triggers a deadlock. In particular, Python's GIL may be held when
    // calling the allocator so it is unsafe to try to acquire the GIL in this
    // callback.
    virtual void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) = 0;
    // start from torch 2.7, not support input 'allocator' in this version
    // will gradually fill in missing abilities
    virtual void createOrIncrefPool(
        c10::DeviceIndex device,
        MempoolId_t mempool_id,
        std::shared_ptr<NPUAllocator> allocator = nullptr)
    {
        TORCH_CHECK(
            false,
            name(),
            " does not yet support createOrIncrefPool. "
            "If you need it, please file an issue describing your use case.");
    }
    using c10::DeviceAllocator::emptyCache;
};

// Allocator object, statically initialized
// See BackendInitializer in CUDACachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
C10_NPU_API extern std::atomic<NPUAllocator*> allocator;


inline NPUAllocator* get()
{
    return allocator.load();
}

inline c10::DataPtr allocate_with_aligned(size_t size, size_t base_addr_aligned_kb)
{
    return get()->allocate_with_aligned(size, base_addr_aligned_kb);
}

// Called directly by clients.
inline void* raw_alloc(size_t nbytes)
{
    return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream)
{
    return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr)
{
    return get()->raw_delete(ptr);
}

inline void init()
{
    uint32_t device_count = 0;
    NPU_CHECK_ERROR(aclrtGetDeviceCount(&device_count));
    return get()->init(device_count);
}

inline double getMemoryFraction(int device)
{
    return get()->getMemoryFraction(device);
}

inline void setMemoryFraction(double fraction, int device)
{
    return get()->setMemoryFraction(fraction, device);
}

inline void emptyCacheImpl(bool check_error = true, bool free_physical = true)
{
    return get()->emptyCacheImpl(check_error, free_physical);
}

// overload, align to cuda, not outer(C10_NPU_API) in this version
inline void emptyCache(MempoolId_t mempool_id)
{
    return get()->emptyCache(mempool_id);
}

C10_NPU_API inline void emptyCache(bool check_error = true)
{
    return get()->emptyCache(check_error);
}

C10_NPU_API inline void emptyVirtAddrCache(bool check_error = true)
{
    return get()->emptyVirtAddrCache(check_error);
}

inline void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock)
{
    return get()->cacheInfo(dev_id, cachedAndFree, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size)
{
    return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream)
{
    return get()->recordStream(ptr, stream);
}

inline void eraseStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream)
{
    return get()->eraseStream(ptr, stream);
}

inline void eraseStreamWithBlockPtr(void* block_ptr, c10_npu::NPUStream stream, void* work_ptr)
{
    return get()->eraseStreamWithBlockPtr(block_ptr, stream, work_ptr);
}

inline void* getBlockPtr(const c10::DataPtr& ptr)
{
    return get()->getBlockPtr(ptr);
}

inline void recordHcclWorkForBlock(void* block_ptr, void* work_ptr)
{
    return get()->recordHcclWorkForBlock(block_ptr, work_ptr);
}

inline DeviceStats getDeviceStats(c10::DeviceIndex device)
{
    return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(c10::DeviceIndex device)
{
    return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(c10::DeviceIndex device)
{
    return get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot()
{
    return get()->snapshot();
}

inline std::shared_ptr<AllocatorState> getCheckpointState(
    c10::DeviceIndex device,
    MempoolId_t id)
{
    return get()->getCheckpointState(device, id);
}

inline CheckpointDelta setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<AllocatorState> pps)
{
    return get()->setCheckpointPoolState(device, std::move(pps));
}

// CUDAGraph interactions
inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(aclrtStream)> filter)
{
    get()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
{
    get()->endAllocateToPool(device, mempool_id);
}

inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id)
{
    return get()->releasePool(device, mempool_id);
}

inline bool hasCapturesUnderway(c10::DeviceIndex device)
{
    return get()->hasCapturesUnderway(device);
}

inline std::shared_ptr<void> getIpcDevPtr(std::string handle)
{
    return get()->getIpcDevPtr(handle);
}

inline ShareableHandle shareIpcHandle(void* ptr)
{
    return get()->shareIpcHandle(ptr);
}

inline void FreeDeviceCachedMemory(int device)
{
    return get()->FreeDeviceCachedMemory(device);
}

inline std::string name()
{
    return get()->name();
}

inline void recordHistory(bool enabled, CreateContextFn context_recorder,
                          size_t alloc_trace_max_entries, RecordContext when)
{
    return get()->recordHistory(enabled, context_recorder,
                                alloc_trace_max_entries, when);
}

inline bool isHistoryEnabled()
{
    return get()->isHistoryEnabled();
}

inline bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations)
{
    return get()->checkPoolLiveAllocations(device, mempool_id, expected_live_allocations);
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer)
{
    return get()->attachOutOfMemoryObserver(observer);
}

inline void attachOomRejectionObserver(OomRejectionObserver observer)
{
    return get()->attachOomRejectionObserver(std::move(observer));
}

inline void attachAllocatorTraceTracker(AllocatorTraceTracker tracker)
{
    return get()->attachAllocatorTraceTracker(std::move(tracker));
}

inline bool checkUceInMemPool(int device)
{
    return get()->checkUceInMemPool(device);
}

inline bool checkBlockIsSafe(const c10::DataPtr& ptr)
{
    return get()->checkBlockIsSafe(ptr);
}

inline void markAllBlockUnsafe(int device)
{
    return get()->markAllBlockUnsafe(device);
}

inline void updateBlockToSafe(const c10::DataPtr& ptr)
{
    return get()->updateBlockToSafe(ptr);
}

inline void cleanEvent()
{
    return get()->cleanEvent();
}

inline void buildServerMemMapForHccl(int device, std::shared_ptr<c10d_npu::HCCLComm> hcclComm)
{
    return get()->buildServerMemMapForHccl(device, hcclComm);
}

inline void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::shared_ptr<NPUAllocator> allocator = nullptr)
{
    get()->createOrIncrefPool(device, mempool_id, std::move(allocator));
}

inline int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id)
{
    return get()->getPoolUseCount(device, mempool_id);
}

bool checkConfigExpandableSegments();

bool isConfig1GPageSizeEnable();

C10_NPU_API void setAllocatorSettings(const std::string& settings);

C10_NPU_API bool saveDevMemUsageInfo(const int& device);

} // namespace NPUCachingAllocator
} // namespace c10_npu
