#include <ATen/DeviceGuard.h>
#include <c10/core/thread_pool.h>
#include <future>

#include <c10/core/DeviceGuard.h>
#include <c10/util/llvmMathExtras.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <c10/util/Logging.h>
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

#ifndef BUILD_LIBTORCH
#include <Python.h>
#endif

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <c10/util/irange.h>

#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"

#include <c10/util/flat_hash_map.h>

using Block = at::HostBlock<c10_npu::NPUStream>;

namespace at_npu::native {
constexpr size_t kMinBlockSize = 512;                 // all sizes are rounded to at least 512 bytes
constexpr size_t segmentSize = 20971520;              // segment size 20M
constexpr size_t deviceTotalSize = 68719476736;       // 64 GB

struct ExpandableBlock;
using Comparison = bool (*)(const ExpandableBlock *, const ExpandableBlock *);
static bool BlockComparatorSize(const ExpandableBlock *a, const ExpandableBlock *b);
static bool BlockComparatorAddress(const ExpandableBlock *a, const ExpandableBlock *b);

struct BlockPool {
    std::set<ExpandableBlock *, Comparison> blocks;
    std::set<ExpandableBlock *, Comparison> unmapped;
    BlockPool() : blocks(BlockComparatorSize), unmapped(BlockComparatorAddress) {
    }
};

struct ExpandableSegment;

struct ExpandableBlock : at::HostBlock<c10_npu::NPUStream>{
    size_t requested_size{};  // memory originally requested
    BlockPool *pool{};        // owning memory pool
    bool mapped{ true };    // is the virtual address range this Block references
                            // backed by physical pages. Always true when
                            // expandable_segment_ is null. When false
                            // This Block will be aligned to the segment size
                            // of its expandable_segment_.
    ExpandableBlock *prev{};            // prev block if split from a larger allocation
    ExpandableBlock *next{};            // next block if split from a larger allocation
    ExpandableSegment *expandable_segment_{ nullptr };
    ExpandableBlock(const size_t size, void *ptr)
        : HostBlock(size, ptr),
          prev(nullptr),
          next(nullptr)
        {}

    ExpandableBlock(const size_t size, BlockPool *pool, void *ptr)
        : HostBlock(size, ptr),
          pool(pool),
          prev(nullptr),
          next(nullptr)
    {}

    // constructor for search key
    explicit ExpandableBlock(const size_t size)
        : HostBlock(size),
          prev(nullptr),
          next(nullptr)
    {}

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }
    void splice(ExpandableBlock *before, ExpandableBlock *after)
    {
        if (before) {
            TORCH_INTERNAL_ASSERT(before->next == after, PTA_ERROR(ErrCode::PTR));
            before->next = this;
        }
        prev = before;
        if (after) {
            TORCH_INTERNAL_ASSERT(after->prev == before, PTA_ERROR(ErrCode::PTR));
            after->prev = this;
        }
        next = after;
    }
};

void innerInitNPU()
{
    // check whether we need std::call_once here
    C10_LOG_API_USAGE_ONCE("aten.init.npu");
    c10_npu::NpuSysCtrl::SysStatus status =
        c10_npu::NpuSysCtrl::GetInstance().Initialize();
    if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
        ASCEND_LOGE("Npu init fail.");
    }
}

class EventPool {
public:
    using Event = std::unique_ptr<
        c10_npu::NPUEvent,
        std::function<void(c10_npu::NPUEvent*)>>;
    EventPool() : pools_(c10_npu::device_count()) {}

    Event get(at::DeviceIndex device)
    {
        TORCH_INTERNAL_ASSERT(0 <= device, PTA_ERROR(ErrCode::PARAM));
        TORCH_INTERNAL_ASSERT(device < static_cast<at::DeviceIndex>(pools_.size()), PTA_ERROR(ErrCode::PARAM));
        auto& pool = pools_[device];
        auto destructor = [&pool](c10_npu::NPUEvent* event) {
            std::lock_guard<std::mutex> g(pool.mutex_);
            pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
        };

        // Try to acquire an event from the per-device pool.
        {
            std::lock_guard<std::mutex> g(pool.mutex_);
            if (!pool.event_pool_.empty()) {
                auto* event = pool.event_pool_.back().release();
                pool.event_pool_.pop_back();
                return Event(event, destructor);
            }
        }
        // otherwise, allocate a new event that will be returned to the pool on
        // destruction.
        return Event(
            std::make_unique<c10_npu::NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS).release(),
            destructor);
    }

    void empty_cache()
    {
        for (auto& pool : pools_) {
            std::lock_guard<std::mutex> g(pool.mutex_);
            pool.event_pool_.clear();
        }
    }

private:
    struct PerDevicePool {
        alignas(64) std::mutex mutex_;
        std::vector<std::unique_ptr<c10_npu::NPUEvent>> event_pool_;
    };
    std::vector<PerDevicePool> pools_;
};

struct NPUCachingHostAllocatorImpl : public at::CachingHostAllocatorImpl<c10_npu::NPUStream, EventPool::Event, ExpandableBlock> {
public:
    virtual bool ptr_check(void* ptr)
    {
        std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
        return npu_ptrs_.find(ptr) != npu_ptrs_.end();
    }

    virtual at::HostStats getHostStats()
    {
        return getStats();
    }

    virtual void resetHostAccumulatedStats()
    {
        resetAccumulatedStats();
    }

    virtual void resetHostPeakStats()
    {
        resetPeakStats();
    }

private:
    void allocate_host_memory(size_t size, void** ptr) override
    {
        // alloc needs set device first when using dataloader with pin_memory=True
        if (c10_npu::GetLocalDevice() < 0) {
            c10_npu::SetCurrentDevice();
        }

        bool host_registered = false;
        auto start = std::chrono::steady_clock::now();
        static bool va_feature_support = true;
        if (c10_npu::acl::AclrtMallocHostWithCfgExist() && va_feature_support) {
            aclrtMallocAttrValue attrValue;
            attrValue.vaFlag = 1;

            aclrtMallocAttribute attributes[1];
            attributes[0].attr = ACL_RT_MEM_ATTR_VA_FLAG;
            attributes[0].value = attrValue;

            aclrtMallocConfig cfg;
            cfg.numAttrs = 1;
            cfg.attrs = attributes;

            aclError mallocError = c10_npu::acl::AclrtMallocHostWithCfg(ptr, size, &cfg);
            bool pinned_mem_register = c10_npu::NPUCachingAllocator::CachingAllocatorConfig::pinned_mem_register();
            // if feature not support, then fall back to the old logic
            if (ACL_ERROR_RT_FEATURE_NOT_SUPPORT == mallocError) {
                va_feature_support = false;
                if (pinned_mem_register) {
                    TORCH_NPU_WARN_ONCE("The pinned_mem_register configuration does not take effect, the current driver version does not support this feature."
                    "To use this feature, you need to upgrade to version 25.5.0 or higher");
                }
                NPU_CHECK_ERROR(aclrtMallocHost(ptr, size), "aclrtMallocHost");
            } else {
                NPU_CHECK_ERROR(mallocError, "aclrtMallocHostWithCfg");
                if (pinned_mem_register) {
                    NPU_CHECK_ERROR(c10_npu::acl::AclrtHostRegisterV2(*ptr, size, ACL_HOST_REG_MAPPED), "aclrtHostRegister failed.");
                    host_registered = true;
                }
            }
        } else {
            aclError err = aclrtMallocHost(ptr, size);
            if (err != ACL_ERROR_NONE) {
                CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
            }
        }

        if (*ptr != nullptr) { // we add the segment pointer here when initialization, but it does not matter
            std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
            npu_ptrs_.insert(*ptr);
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(*ptr) == 0);
            use_host_register[*ptr] = host_registered;
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        {
            std::lock_guard<std::mutex> g(stats_.timing_mutex_);
            stats_.host_alloc_time.increase(duration.count());
        }
    }

    void free_block(ExpandableBlock* block) override
    {
        auto start = std::chrono::steady_clock::now();
        void* ptr = block->ptr_;
        bool use_register = false;
        {
            std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(ptr) == 1);
            use_register = use_host_register[ptr];
        }
        if (use_register) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtHostUnregister(ptr), "aclrtHostUnregister");
        }
        aclError err = aclrtFreeHost(block->ptr_);
        if (err != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
        }
        if (ptr != nullptr) {
            std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
            use_host_register.erase(ptr);
            npu_ptrs_.erase(block->ptr_);
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        {
            std::lock_guard<std::mutex> g(stats_.timing_mutex_);
            stats_.host_free_time.increase(duration.count());
        }
    }

    void record_stream(std::optional<std::vector<EventPool::Event>>& events, c10_npu::NPUStream stream) override
    {
        if (c10_npu::NpuSysCtrl::GetInstance().GetHostFinalize()) {
            return;
        }
        auto event = create_event_internal(stream.device_index());
        event->record(stream);
        events->push_back(std::move(event));
    }

    bool query_event(EventPool::Event& event) override
    {
        bool isEventCompleted = true;
        try {
            // Query the completion status of the NPU event here. In some runtime scenarios (for example, HBM error),
            // querying the event may throw an exception. In this case, we need to pop the event from the event queue to
            // avoid coredump.
            isEventCompleted = event->query();
        } catch (const std::exception& e) {
            ASCEND_LOGE("query_event() failed, runtime error: %s", e.what());
            isEventCompleted = true;
        } catch (...) {
            ASCEND_LOGE("query_event() failed with unknown error!");
            isEventCompleted = true;
        }
        return isEventCompleted;
    }

    EventPool::Event create_event_internal(at::DeviceIndex idx)
    {
        static auto* event_pool = new EventPool();
        return event_pool->get(idx);
    }

private:
    std::mutex npu_ptrs_mutex_;
    ska::flat_hash_set<void*> npu_ptrs_;
    ska::flat_hash_map<void*, bool> use_host_register;
};


static bool BlockComparatorSize(const ExpandableBlock *a, const ExpandableBlock *b)
{
    if (a->size_ != b->size_) {
        return a->size_ < b->size_;
    }
    return reinterpret_cast<uintptr_t>(a->ptr_) < reinterpret_cast<uintptr_t>(b->ptr_);
}

static bool BlockComparatorAddress(const ExpandableBlock *a, const ExpandableBlock *b)
{
    return reinterpret_cast<uintptr_t>(a->ptr_) < reinterpret_cast<uintptr_t>(b->ptr_);
}

struct SegmentRange {
    char *ptr;
    size_t size;
    SegmentRange(void *p, size_t s) : ptr(static_cast<char *>(p)), size(s) {}
};

struct ExpandableSegment {
    explicit ExpandableSegment(size_t size) : segment_size_(size)
    {
        // we allocate enough address space for 1 1/8 the total memory on the NPU.
        // This allows for some cases where we have to unmap pages earlier in the
        // segment to put them at the end.
        max_handles_ = numSegments(deviceTotalSize + deviceTotalSize / 8);

        NPU_CHECK_ERROR(
            c10_npu::acl::AclrtReserveMemAddress(&ptr_, segment_size_ * max_handles_, 0, nullptr, 1, nullptr));
        ASCEND_LOGD("NPUExpandableHostAllocator malloc by AclrtReserveMemAddress: size=%zu, segment_size=%zu",
            segment_size_ * max_handles_, segment_size_);
    }

    // begin must be aligned to segment_size_.
    // returns the actual range mapped, which may be
    // greater than requested if size is not aligned to segment_size_.
    // return size of 0 indicates OOM
    SegmentRange map(SegmentRange range, BlockPool *pool)
    {
        auto begin = segmentLeft(range.ptr);
        auto end = segmentRight(range.ptr + range.size);
        TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr, PTA_ERROR(ErrCode::PTR));
        if (begin == end) {
            return rangeFromHandles(begin, end);
        }
        while (end > handles_.size()) {
            handles_.emplace_back(std::nullopt);
        }
        for (auto i : c10::irange(begin, end)) {
            TORCH_INTERNAL_ASSERT(!handles_.at(i), PTA_ERROR(ErrCode::VALUE));
            aclrtDrvMemHandle handle = nullptr;
            aclrtPhysicalMemProp prop = {};
            prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr = ACL_HBM_MEM_HUGE;
            prop.location.type = ACL_MEM_LOCATION_TYPE_HOST;
            prop.location.id = 0;
            prop.reserve = 0;
            ASCEND_LOGD("Alloc memory from physical host for block %zu", i);
            auto status = c10_npu::acl::AclrtMallocPhysical(&handle, segment_size_, &prop, 0);
            if (status == ACL_ERROR_RT_MEMORY_ALLOCATION) {
                for (auto j : c10::irange(begin, i)) {
                    auto h = handles_.at(j).value();
                    handles_.at(j) = std::nullopt;
                    NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h.handle));
                }
                trimHandles();
                return rangeFromHandles(begin, begin);
            }
            NPU_CHECK_ERROR(status, "aclrtMallocPhysical");
            handles_.at(i) = Handle{handle};
        }
        for (auto i : c10::irange(begin, end)) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtMapMem(static_cast<char *>(ptr_) + i * segment_size_, segment_size_, 0,
                handles_.at(i).value().handle, 0, nullptr));
        }
        ASCEND_LOGD("NPUExpandableHostAllocator map: segment_size=%zu", segment_size_);
        return rangeFromHandles(begin, end);
    }

    // unmaps all the completely empty segment_size_ segments between
    // [begin, begin + size), returns the offset where the range begin,
    // and the actual size unmapped (multiple of segment_size_)
    SegmentRange unmap(const SegmentRange range)
    {
        auto begin = segmentRight(range.ptr);
        auto end = segmentLeft(range.ptr + range.size);
        if (begin >= end) {
            return SegmentRange{ range.ptr, 0 };
        }
        unmapHandles(begin, end);
        return rangeFromHandles(begin, end);
    }

    char *ptr() const
    {
        return static_cast<char *>(ptr_);
    }

    size_t size() const
    {
        return max_handles_ * segment_size_;
    }

    ~ExpandableSegment()
    {
        forEachAllocatedRange([&](size_t begin, size_t end) { unmapHandles(begin, end); });
        NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr_, nullptr));
        ASCEND_LOGD("NPUExpandableHostAllocator free by AclrtReleaseMemAddress");
    }

private:
    void unmapHandles(size_t begin, size_t end)
    {
        for (auto i : c10::irange(begin, end)) {
            Handle h = handles_.at(i).value();
            handles_.at(i) = std::nullopt;
            NPU_CHECK_ERROR(c10_npu::acl::AclrtUnmapMem(static_cast<char *>(ptr_) + segment_size_ * i, nullptr));
            NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h.handle));
        }
        ASCEND_LOGD("NPUExpandableHostAllocator unmap: segment_size=%zu", segment_size_);
        trimHandles();
    }

    void trimHandles()
    {
        while (!handles_.empty() && !handles_.back()) {
            handles_.pop_back();
        }
    }

    void forEachAllocatedRange(const std::function<void(size_t, size_t)> fn) const
    {
        size_t start = 0;
        for (auto i : c10::irange(handles_.size())) {
            if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
                start = i;
            }
            if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
                fn(start, i + 1);
            }
        }
    }

    size_t numSegments(const size_t size) const
    {
        return (size + segment_size_ - 1) / segment_size_;
    }

    size_t segmentLeft(const char *p) const
    {
        auto size = p - ptr();
        return static_cast<size_t>(size) / segment_size_;
    }

    size_t segmentRight(const char *p) const
    {
        auto size = p - ptr();
        return numSegments(size);
    }

    SegmentRange rangeFromHandles(size_t begin, size_t end) const
    {
        return {ptr() + segment_size_ * begin, segment_size_ * (end - begin)};
    }

    void *ptr_{};
    size_t max_handles_{};
    size_t segment_size_{};
    struct Handle {
        aclrtDrvMemHandle handle;
    };
    std::vector<std::optional<Handle>> handles_;
};

struct AllocParams {
    AllocParams(size_t size, BlockPool *pool)
        : search_key(size),
          pool(pool),
          err(ACL_ERROR_NONE)
    {}

    [[nodiscard]] size_t size() const
    {
        return search_key.size_;
    }

    ExpandableBlock search_key;

    BlockPool *pool;

    ExpandableBlock *block{};

    aclError err;
};

struct NPUExpandableHostAllocatorImpl : public NPUCachingHostAllocatorImpl {
public:
    std::pair<void*, void*> allocate(size_t orig_size) override
    {
        if (orig_size == 0) {
            return {nullptr, nullptr};
        }

        std::lock_guard<std::mutex> lock(mutex);

        process_events();

        const auto size = round_size(orig_size);

        AllocParams params(size, &blocks_pool);

        // First, try to get a block from the existing pool.
        // If the block cannot be got from pool.blocks, try to get a block from the pool.unmapped.
        bool block_found = get_free_expandable_block(params) || alloc_block(params);
        if (!block_found) {
            ASCEND_LOGE("Get a block from the existing pool failed. Try to free cached blocks and reallocate. This error log can be ignored.");
            block_found = release_cached_blocks() && alloc_block(params);
        }

        if (!block_found) {
            if (params.err == ACL_ERROR_RT_MEMORY_ALLOCATION) {
                AT_ERROR("Malloc pin memory failed, host out of memory, Rried to allocate size: ", orig_size,
                    ", total_allocated_memory: ", stats.allocated_bytes.current, ", total_reserved_memory: ", stats.reserved_bytes.current,
                    ". You might be able to retry after freeing up memory using torch_npu.npu.host_empty_cache().");
            }
            return {nullptr, nullptr};
        }

        auto found_block = alloc_found_block(params, orig_size);
        //*ptr = found_block->ptr;
        return {found_block->ptr_, reinterpret_cast<void *>(found_block)};
    }

    void free(void *ctx) override
    {
        if (!ctx) {
            return;
        }
        auto* ptr_block = static_cast<ExpandableBlock*>(ctx);
        AT_ASSERT(ptr_block, PTA_ERROR(ErrCode::VALUE));
        std::lock_guard<std::mutex> lock(mutex);
        ExpandableBlock *block = get_active_block(ptr_block->ptr_);
        AT_ASSERT(block, PTA_ERROR(ErrCode::VALUE));
        block_free(block);
    }

    void empty_cache() override
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_cached_blocks();
    }

    bool ptr_check(void *ptr) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        return ptr_to_block_.find(ptr) != ptr_to_block_.end();
    }

    /* * Returns a copy of the memory allocator stats * */
    at::HostStats getHostStats() override
    {
        return stats;
    }

    void resetHostAccumulatedStats() override
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats.allocated_bytes.reset_accumulated();
        stats.reserved_bytes.reset_accumulated();

        stats.host_alloc_time.reset_accumulated();
        stats.host_free_time.reset_accumulated();
    }

    void resetHostPeakStats() override
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats.allocated_bytes.reset_peak();
        stats.reserved_bytes.reset_peak();

        stats.host_alloc_time.reset_peak();
        stats.host_free_time.reset_peak();
    }

    bool record_event(void* ptr, void* ctx, c10_npu::NPUStream s) override
    {
        c10_npu::NPUStream stream = c10_npu::NPUStream(s);

        auto* block = reinterpret_cast<ExpandableBlock*>(ctx);

        // Note: we need to check if the passed-in `ctx` is valid. This is because
        // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
        // an arbitrary tensor, and is not guaranteed to correspond to a pinned
        // memory allocation. Therefore, we need to check that `ctx` is valid before
        // proceeding.
        {
            std::lock_guard<std::mutex> g(blocks_mutex_);
            if (ptr_to_block_.find(block) != ptr_to_block_.end()) {
                // Now we know this object is safe to access.
                std::lock_guard<std::mutex> gb(block->mutex_);
                TORCH_INTERNAL_ASSERT(block->allocated_);
                block->streams_.insert(stream);
                return true;
            }
            auto it = ptr_to_block_.find(ptr);
            if (it != ptr_to_block_.end()) {
                block = it->second;
                std::lock_guard<std::mutex> g(block->mutex_);
                TORCH_INTERNAL_ASSERT(block->allocated_);
                block->streams_.insert(stream);
                return true;
            }
        }
        return false;
    }

private:
    BlockPool blocks_pool;

    std::mutex mutex;

    std::mutex stats_mutex_;

    // allocated or in use by a stream
    ska::flat_hash_map<void *, ExpandableBlock *> ptr_to_block_;

    // all live expandable segments
    std::vector<ExpandableSegment *> expandable_segments_;

    // pin memory statistics
    at::HostStats stats;

    alignas(64) std::mutex blocks_mutex_;

    alignas(64) std::mutex events_mutex_;

    std::deque<std::pair<EventPool::Event, ExpandableBlock*>> events_; // event queue paired with block

    void block_free(ExpandableBlock *block)
    {
        block->allocated_ = false;

        if (block->streams_.empty()) {
            free_expandable_block(block);
            return;
        }

        // Note: we can assume that free is correctly paired with alloc, and thus we
        // do not need to look up the ctx in blocks_.
        std::optional<std::vector<EventPool::Event>> events;
        {
            std::lock_guard<std::mutex> g(block->mutex_);
            block->allocated_ = false;
            if (block->streams_.empty()) {
                TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
            } else {
                events = std::vector<EventPool::Event>();
                events->reserve(block->streams_.size());
                for (const auto& stream : block->streams_) {
                    record_stream(events, c10_npu::NPUStream(stream));
                }
                block->event_count_ += events->size();
                block->streams_.clear();
            }
        }

        if (events) {
            // restore these events that record by used streams.
            std::lock_guard<std::mutex> g(events_mutex_);
            for (auto&& event : *events) {
                events_.emplace_front(std::move(event), block);
            }
        }
    }

    ExpandableBlock *get_active_block(void *ptr)
    {
        auto it = ptr_to_block_.find(ptr);
        if (it == ptr_to_block_.end()) {
            return nullptr;
        }
        ExpandableBlock *block = it->second;
        return block;
    }

    void release_blocks()
    {
        std::vector<ExpandableBlock *> to_unmap;
        auto it = blocks_pool.blocks.begin();
        while (it != blocks_pool.blocks.end()) {
            ExpandableBlock *block = *it;
            ++it;
            if (block->expandable_segment_) {
                // unmapping will mutate the free pool
                // so just gather what needs to be freed
                // to avoid invalidating the iterator
                to_unmap.push_back(block);
            }
        }
        for (ExpandableBlock *block : to_unmap) {
            unmap_block(block);
            if (!block->prev && !block->next) {
                release_expandable_segment(block);
            }
        }
    }

    void unmap_block(ExpandableBlock *block)
    {
        auto start = std::chrono::steady_clock::now();
        auto unmapped = block->expandable_segment_->unmap(SegmentRange{ block->ptr_, block->size_ });
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // Update the statistics on the time spent on AclrtUnmapMem/AclrtFreePhysical
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.host_free_time.increase(duration.count());
        }
        if (unmapped.size == 0) {
            return;
        }
        block->pool->blocks.erase(block);

        ptrdiff_t before_size = static_cast<char *>(unmapped.ptr) - static_cast<char *>(block->ptr_);
        if (before_size > 0) {
            // prev? -> before_free -> block
            ExpandableBlock *before_free = new ExpandableBlock(before_size, block->pool, block->ptr_);
            before_free->expandable_segment_ = block->expandable_segment_;
            before_free->splice(block->prev, block);
            block->pool->blocks.insert(before_free);
        }

        auto after_size = block->size_ - (before_size + unmapped.size);
        if (after_size > 0) {
            // block -> after_free -> next?
            ExpandableBlock *after_free = new ExpandableBlock(after_size, block->pool, unmapped.ptr + unmapped.size);
            after_free->expandable_segment_ = block->expandable_segment_;
            after_free->splice(block, block->next);
            block->pool->blocks.insert(after_free);
        }

        block->ptr_ = unmapped.ptr;
        block->size_ = unmapped.size;
        block->mapped = false;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.reserved_bytes.decrease(unmapped.size);
        }
        try_merge_blocks(block, block->prev, *block->pool);
        try_merge_blocks(block, block->next, *block->pool);
        block->pool->unmapped.insert(block);
    }

    void release_expandable_segment(ExpandableBlock *block)
    {
        TORCH_INTERNAL_ASSERT(block->size_ == block->expandable_segment_->size(), "block disagrees with segment",
            PTA_ERROR(ErrCode::INTERNAL));
        TORCH_INTERNAL_ASSERT(!block->mapped, PTA_ERROR(ErrCode::INTERNAL));
        auto it = std::find(expandable_segments_.begin(), expandable_segments_.end(), block->expandable_segment_);
        TORCH_INTERNAL_ASSERT(it != expandable_segments_.end(), PTA_ERROR(ErrCode::INTERNAL));
        expandable_segments_.erase(it);
        block->pool->unmapped.erase(block);
        delete block->expandable_segment_;
        block->expandable_segment_ = nullptr;
        delete block;
        block = nullptr;
    }

    /* * combine previously split blocks. returns the size of the subsumed block, or 0 on failure. * */
    size_t try_merge_blocks(ExpandableBlock *dst, ExpandableBlock *src, BlockPool &pool)
    {
        if (!src || src->allocated_ || src->event_count_ > 0 || !src->streams_.empty() || dst->mapped != src->mapped) {
            return 0;
        }

        AT_ASSERT(dst->is_split() && src->is_split(), PTA_ERROR(ErrCode::VALUE));

        if (dst->prev == src) {
            dst->ptr_ = src->ptr_;
            dst->prev = src->prev;
            if (dst->prev) {
                dst->prev->next = dst;
            }
        } else {
            dst->next = src->next;
            if (dst->next) {
                dst->next->prev = dst;
            }
        }

        const size_t subsumed_size = src->size_;
        dst->size_ += subsumed_size;
        auto erased = src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
        delete src;
        src = nullptr;

        return subsumed_size;
    }

    /* * moves a block into a pool of cached free blocks * */
    void free_expandable_block(ExpandableBlock *block)
    {
        AT_ASSERT(!block->allocated_ && block->event_count_ == 0, PTA_ERROR(ErrCode::VALUE));

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.allocated_bytes.decrease(block->size_);
        }
        auto &pool = *block->pool;
        const std::array<ExpandableBlock *, 2> merge_candidates = { block->prev, block->next };
        for (ExpandableBlock *merge_candidate : merge_candidates) {
            try_merge_blocks(block, merge_candidate, pool);
        }
        ptr_to_block_.erase(block->ptr_);
        pool.blocks.insert(block);
    }

    void process_events() override
    {
        while (true) {
            // Avoid calling cudaEventDestroy while holding a mutex, so move
            // intermediate events out of the lock into this object.
            // process the last event
            std::optional<std::pair<EventPool::Event, ExpandableBlock*>> processed;
            {
                std::lock_guard<std::mutex> g(events_mutex_);
                if (!events_.empty()) {
                    processed = std::move(events_.back());
                    events_.pop_back();
                }
            }

            if (!processed) {
                return;
            }
            auto& event = processed->first;
            bool isEventCompleted = true;
            try {
                isEventCompleted = event->query();
            } catch (...) {
                ASCEND_LOGE("process_events() query event failed!");
                // event query failed, pop the event
                isEventCompleted = true;
            }
            if (!isEventCompleted) {
                {
                    std::lock_guard<std::mutex> g(events_mutex_);
                    events_.push_back(std::move(*processed));
                    return;
                }
            }

            TORCH_INTERNAL_ASSERT(processed);
            auto* block = processed->second;
            {
                std::lock_guard<std::mutex> g(block->mutex_);
                block->event_count_--;
                if (block->event_count_ == 0 && !block->allocated_) {
                    free_expandable_block(block);
                }
            }
        }
    }

    void record_stream(std::optional<std::vector<EventPool::Event>>& events, c10_npu::NPUStream stream) override
    {
        static auto* event_pool = new EventPool();
        auto event = event_pool->get(stream.device_index());
        event->record(stream);
        events->push_back(std::move(event));
    }

    bool query_event(EventPool::Event& event) override
    {
        bool isEventCompleted = true;
        try {
            // Query the completion status of the NPU event here. In some runtime scenarios (for example, HBM error),
            // querying the event may throw an exception. In this case, we need to pop the event from the event queue to
            // avoid coredump.
            isEventCompleted = event->query();
        } catch (const std::exception& e) {
            ASCEND_LOGE("query_event() failed, runtime error: %s", e.what());
            isEventCompleted = true;
        } catch (...) {
            ASCEND_LOGE("query_event() failed with unknown error!");
            isEventCompleted = true;
        }
        return isEventCompleted;
    }

    static size_t round_size(size_t size)
    {
        size = size + 32;
        if (size < kMinBlockSize) {
            return kMinBlockSize;
        }

        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }

    bool get_free_expandable_block(AllocParams &p)
    {
        BlockPool &pool = *p.pool;
        auto it = pool.blocks.lower_bound(&p.search_key);
        if (it == pool.blocks.end()) {
            return false;
        }

        if ((*it)->expandable_segment_) {
            // if we are allocated to the part of the block that is expandable
            // for the purposes of "best fit" we consider its size to be the size it
            // can expand to, not the size it currently is. This means that we
            // sometimes have to search for blocks with bigger 'size' before
            // choosing this segment.
            auto expandable_size = [](ExpandableBlock *b) {
                return b->size_ + (b->next && !b->next->mapped ? b->next->size_ : 0);
            };
            auto next = it;
            next++;
            while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
                expandable_size(*next) < expandable_size(*it)) {
                it = next++;
            }
        }

        p.block = *it;
        pool.blocks.erase(it);
        return true;
    }

    bool alloc_block(AllocParams &p)
    {
        p.block = try_allocate_expandable_block(p.pool, p.size());
        if (p.block) {
            p.err = ACL_ERROR_NONE;
        } else {
            p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
        }
        return static_cast<bool>(p.block);
    }

    // returns the smallest possible address in any segment
    // where there is enough free address space to fit size
    // may be composed of free and unmapped segments
    ExpandableBlock *find_expandable_block(BlockPool *pool, size_t size)
    {
        ExpandableBlock key(0);

        auto allocatable = [](Block *b) {
            return b && !b->allocated_ && b->event_count_ == 0 && b->streams_.empty();
        };

        auto has_available_address_space = [&](ExpandableBlock *b) {
            size_t bytes = 0;
            while (bytes < size && allocatable(b)) {
                bytes += b->size_;
                b = b->next;
            }
            return bytes >= size;
        };
        for (auto it = pool->unmapped.lower_bound(&key); it != pool->unmapped.end(); ++it) {
            ExpandableBlock *c = *it;
            // we found the lowest address of an unmapped segment
            // but there might be a free segment we can also use
            // right before it
            if (allocatable(c->prev)) {
                c = c->prev;
            }
            if (has_available_address_space(c)) {
                return c;
            }
        }

        auto segment = new (std::nothrow) ExpandableSegment(segmentSize);
        if (!segment) {
            ASCEND_LOGE("Failed to allocate ExpandableSegment.");
            return nullptr;
        }

        expandable_segments_.emplace_back(segment);

        ExpandableSegment *es = expandable_segments_.back();
        ExpandableBlock *candidate = new (std::nothrow) ExpandableBlock(es->size(), pool, es->ptr());
        if (!candidate) {
            ASCEND_LOGE("Failed to allocate Block.");
            return nullptr;
        }
        candidate->mapped = false;
        candidate->expandable_segment_ = es;
        pool->unmapped.insert(candidate);
        return candidate;
    }

    bool map_block(ExpandableBlock *to_map, size_t size, BlockPool *map_pool)
    {
        TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size_, PTA_ERROR(ErrCode::VALUE));

        auto mapped_range = to_map->expandable_segment_->map(SegmentRange{ to_map->ptr_, size }, map_pool);
        // failed to map the memory
        if (mapped_range.size == 0) {
            return false;
        }
        TORCH_INTERNAL_ASSERT(mapped_range.ptr == to_map->ptr_ && mapped_range.size >= size,
            PTA_ERROR(ErrCode::INTERNAL));

        BlockPool &pool = *to_map->pool;
        pool.unmapped.erase(to_map);
        to_map->mapped = true;

        if (mapped_range.size < to_map->size_) {
            // to_map -> remaining -> to_map->next(?)
            ExpandableBlock *remaining = new ExpandableBlock(to_map->size_ - mapped_range.size, &pool,
                static_cast<char *>(to_map->ptr_) + mapped_range.size);
            remaining->mapped = false;
            remaining->expandable_segment_ = to_map->expandable_segment_;
            remaining->splice(to_map, to_map->next);
            pool.unmapped.insert(remaining);
            to_map->size_ = mapped_range.size;
        }

        try_merge_blocks(to_map, to_map->prev, pool);
        try_merge_blocks(to_map, to_map->next, pool);
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.reserved_bytes.increase(mapped_range.size);
        }
        pool.blocks.insert(to_map);
        return true;
    }

    ExpandableBlock *try_allocate_expandable_block(BlockPool *pool, size_t size)
    {
        ExpandableBlock *candidate = find_expandable_block(pool, size);
        // Candidate is now a list free/unmapped blocks with at least size room:
        // unmapped -> null
        // unmapped -> free -> *
        // free -> unmapped -> *

        auto start = std::chrono::steady_clock::now();
        if (!candidate->mapped && !map_block(candidate, std::min(candidate->size_, size), pool)) {
            return nullptr;
        }
        TORCH_INTERNAL_ASSERT(candidate->mapped, PTA_ERROR(ErrCode::INTERNAL));

        while (candidate->size_ < size) {
            // invariant: free -> unmapped -> *
            // map_block will map some of unmapped and merge with free
            auto remaining = size - candidate->size_;
            auto new_candidate = candidate->next;
            if (C10_UNLIKELY(new_candidate == nullptr)) {
                return nullptr;
            }
            if (!map_block(new_candidate, std::min(remaining, candidate->next->size_), pool)) {
                return nullptr;
            }
            candidate = new_candidate;
        }
        pool->blocks.erase(candidate);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // Update the statistics on the time spent on AclrtMallocPhysical/AclrtMapMem
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.host_alloc_time.increase(duration.count());
        }
        return candidate;
    }

    ExpandableBlock *alloc_found_block(const AllocParams &params, const size_t orig_size)
    {
        auto size = params.size();
        auto pool = params.pool;

        TORCH_INTERNAL_ASSERT(params.err == ACL_ERROR_NONE && params.block != nullptr && params.block->ptr_ != nullptr,
            PTA_ERROR(ErrCode::PTR));
        ExpandableBlock *block = params.block;
        ExpandableBlock *remaining = nullptr;

        if (params.block->size_ - params.size() >= kMinBlockSize) {
            remaining = block;

            block = new ExpandableBlock(size, pool, block->ptr_);
            block->expandable_segment_ = remaining->expandable_segment_;
            block->prev = remaining->prev;
            if (block->prev) {
                block->prev->next = block;
            }
            block->next = remaining;

            remaining->prev = block;
            remaining->ptr_ = static_cast<char *>(remaining->ptr_) + size;
            remaining->size_ -= size;
            pool->blocks.insert(remaining);
        }

        block->allocated_ = true;
        block->requested_size = orig_size;
        ptr_to_block_[block->ptr_] = block;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.allocated_bytes.increase(block->size_);
        }
        return block;
    }

    // npuSynchronizeDevice must be executed before this function can be called
    bool release_cached_blocks()
    {
        // First ensure that all blocks that can't currently be allocated due to
        // outstanding events are returned to the pool.
        process_events();

        // Free all non-split cached blocks
        release_blocks();
        return true;
    }
};

// Note : we do not use the macro DECLARE_HOST_ALLOCATOR here, because we need to access caching_host_allocator with ptr_exist function
void raw_local_deleter(void* ptr);

struct NPUCachingHostAllocator final
    : public at::CachingHostAllocatorInterface<NPUCachingHostAllocatorImpl> {
    at::DataPtr allocate(size_t size) override
    {
        auto ptr_and_ctx = impl_->allocate(size);
        return {
            ptr_and_ctx.first,
            ptr_and_ctx.second,
            &raw_local_deleter,
            at::DeviceType::CPU};
    }

    void setAllocator(std::unique_ptr<NPUCachingHostAllocatorImpl> hostAllocatorImpl)
    {
        impl_ = std::unique_ptr(std::move(hostAllocatorImpl));
    }
};

static auto cachingHostAllocatorImpl = std::make_unique<NPUCachingHostAllocatorImpl>();
static auto expandableHostAllocatorImpl = std::make_unique<NPUExpandableHostAllocatorImpl>();

static NPUCachingHostAllocator hostAllocator;

static std::once_flag initConfig;

static inline NPUCachingHostAllocator& getNPUCachingHostAllocator()
{
    std::call_once(initConfig, [] {
        if (c10_npu::NPUCachingAllocator::CachingAllocatorConfig::pin_memory_expandable_segments()) {
            hostAllocator.setAllocator(std::move(expandableHostAllocatorImpl));
        } else {
            hostAllocator.setAllocator(std::move(cachingHostAllocatorImpl));
        }
    });

    return hostAllocator;
}

void raw_local_deleter(void* ptr)
{
#ifndef BUILD_LIBTORCH
    // check the current thread have hold GIL Lock.
    if (PyGILState_Check()) {
        // the current thread should not hold GIL.
        Py_BEGIN_ALLOW_THREADS
        getNPUCachingHostAllocator().free(ptr);
        Py_END_ALLOW_THREADS
    } else {
        getNPUCachingHostAllocator().free(ptr);
    }
#else
    getNPUCachingHostAllocator().free(ptr);
#endif
}

bool ptr_exist(void* ptr)
{
    return getNPUCachingHostAllocator().impl_->ptr_check(ptr);
}

at::Allocator* getCachingHostAllocator()
{
    return &getNPUCachingHostAllocator();
}

// the host memory is not allocated by malloc
aclError process_unregistered_mem_location_type(c10_npu::NPUStream stream, aclrtMemcpyKind kind)
{
    // Sync when host memory is allocated by malloc
    ASCEND_LOGD("The copy of the kind needs to be converted to synchronous");
    aclError error = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream);
    if (error != ACL_ERROR_NONE) {
        CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(error);
        C10_NPU_SHOW_ERR_MSG();
        AT_ERROR("ACL stream synchronize failed.");
        return error;
    }

    return ACL_ERROR_NONE;
}

// the host memory is allocated by aclrtMallocHost or malloc and register
void process_host_mem_location_type(const c10::Storage& storage, c10_npu::NPUStream stream)
{
    ASCEND_LOGD("The memory is registered.");
    if (at_npu::native::ptr_exist(storage.mutable_data())) {
        ASCEND_LOGD("The ptr is allocated by torch_npu, then need to record stream.");
        CachingHostAllocator_recordEvent(storage.mutable_data(), storage.data_ptr().get_context(), stream);
    }
}

// process non_blocking copy between host and device
void process_non_blocking_copy(const c10::Storage& storage, void *currentPtr, c10_npu::NPUStream stream, aclrtMemcpyKind kind)
{
    if (c10_npu::acl::AclrtPointerGetAttributesExist()) {
        aclrtPtrAttributes attributes;
        NPU_CHECK_ERROR(c10_npu::acl::AclrtPointerGetAttributes(currentPtr, &attributes), "aclrtPointerGetAttributes");
        ASCEND_LOGD("The ptr is %p, and currentPtr is %p", storage.mutable_data(), currentPtr);
        aclrtMemLocationType ptrType = attributes.location.type;
        ASCEND_LOGD("The aclrtMemType of currentPtr(%p) is %d", currentPtr, ptrType);
        if (ptrType == ACL_MEM_LOCATION_TYPE_HOST) {
            process_host_mem_location_type(storage, stream);
        } else {
            NPU_CHECK_ERROR(process_unregistered_mem_location_type(stream, kind), "aclrtSynchronizeStreamWithTimeout");
        }
    } else {
        ASCEND_LOGD("The AclrtPointerGetAttributes func does not exist.")
        if (at_npu::native::ptr_exist(storage.mutable_data())) {
            ASCEND_LOGD("The ptr is allocated by torch_npu, then need to record stream.");
            CachingHostAllocator_recordEvent(storage.mutable_data(), storage.data_ptr().get_context(), stream);
        } else {
            NPU_CHECK_ERROR(process_unregistered_mem_location_type(stream, kind), "aclrtSynchronizeStreamWithTimeout");
        }
    }
}

c10::Allocator* getPinnedMemoryAllocator()
{
    innerInitNPU();
    return getCachingHostAllocator();
}

void CachingHostAllocator_emptyCache()
{
    getNPUCachingHostAllocator().empty_cache();
}

bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, c10_npu::NPUStream stream)
{
    return getNPUCachingHostAllocator().record_event(ptr, ctx, stream);
}

at::HostStats CachingHostAllocator_getStats()
{
    return getNPUCachingHostAllocator().impl_->getHostStats();
}

void CachingHostAllocator_resetAccumulatedStats()
{
    getNPUCachingHostAllocator().impl_->resetHostAccumulatedStats();
}

void CachingHostAllocator_resetPeakStats()
{
    getNPUCachingHostAllocator().impl_->resetHostPeakStats();
}

} // namespace at_npu::native