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

#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"

#include <ATen/core/CachingHostAllocator.h>
#include <c10/util/flat_hash_map.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"

using Block = at::HostBlock<c10_npu::NPUStream>;

namespace at_npu::native {

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

struct NPUCachingHostAllocatorImpl : public at::CachingHostAllocatorImpl<c10_npu::NPUStream, EventPool::Event> {
public:
    bool ptr_check(void* ptr)
    {
        std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
        return npu_ptrs_.find(ptr) != npu_ptrs_.end();
    }

private:
    void allocate_host_memory(size_t size, void** ptr) override
    {
        // alloc needs set device first when using dataloader with pin_memory=True
        if (c10_npu::GetLocalDevice() < 0) {
            c10_npu::SetCurrentDevice();
        }

        if (get_reserve_segment().initialized()) {
            *ptr = get_reserve_segment().allocate(size);
            if (*ptr != nullptr) {
                std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
                npu_ptrs_.insert(*ptr);
                return;
            }
        }

        allocate_host_memory_slowpath(size, ptr);
    }

    void allocate_host_memory_slowpath(size_t size, void** ptr)
    {
        auto start = std::chrono::steady_clock::now();
        aclError err = aclrtMallocHost(ptr, size);
        if (err != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
        }
        if (*ptr != nullptr) { // we add the segment pointer here when initialization, but it does not matter
            std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
            npu_ptrs_.insert(*ptr);
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        {
            std::lock_guard<std::mutex> g(stats_.timing_mutex_);
            stats_.host_alloc_time.increase(duration.count());
        }
    }

    void free_block(Block* block) override
    {
        if (get_reserve_segment().initialized()) {
            if (get_reserve_segment().owns(block->ptr_)) {
                std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
                npu_ptrs_.erase(block->ptr_);
                return;
            }
        }
        free_block_slowpath(block);
    }

    void free_block_slowpath(Block* block)
    {
        auto start = std::chrono::steady_clock::now();
        void* ptr = block->ptr_;
        aclError err = aclrtFreeHost(block->ptr_);
        if (err != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
        }
        if (ptr != nullptr) {
            std::lock_guard<std::mutex> g(npu_ptrs_mutex_);
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
        auto event = create_event_internal(stream.device_index());
        event->record(stream);
        events->push_back(std::move(event));
    }

    bool query_event(EventPool::Event& event) override
    {
        return event->query();
    }

    EventPool::Event create_event_internal(at::DeviceIndex idx)
    {
        static auto* event_pool = new EventPool();
        return event_pool->get(idx);
    }

    at::PinnedReserveSegment& get_reserve_segment()
    {
        static auto reserve_segment = [&]() {
            if (c10_npu::NPUCachingAllocator::NPUAllocatorConfig::pinned_reserve_segment_size_mb() > 0) {
                void *ptr;
                size_t temp = c10_npu::NPUCachingAllocator::NPUAllocatorConfig::pinned_reserve_segment_size_mb();
                size_t sz = temp * 1024 * 1024; // temp MB
                allocate_host_memory_slowpath(sz, &ptr);
                return at::PinnedReserveSegment(ptr, sz);
            } else {
                return at::PinnedReserveSegment();
            }
        }();
        return reserve_segment;
    }

private:
    std::mutex npu_ptrs_mutex_;
    ska::flat_hash_set<void*> npu_ptrs_;
};

// Note : we do not use the macro DECLARE_HOST_ALLOCATOR here, because we need to access caching_host_allocator with ptr_exist function
void raw_local_deleter(void* ptr);

struct NPUCachingHostAllocator final : public at::CachingHostAllocatorInterface<NPUCachingHostAllocatorImpl, raw_local_deleter> {};

static NPUCachingHostAllocator caching_host_allocator;

void raw_local_deleter(void* ptr)
{
#ifndef BUILD_LIBTORCH
    // check the current thread have hold GIL Lock.
    if (PyGILState_Check()) {
        // the current thread should not hold GIL.
        Py_BEGIN_ALLOW_THREADS
        caching_host_allocator.free(ptr);
        Py_END_ALLOW_THREADS
    } else {
        caching_host_allocator.free(ptr);
    }
#else
    caching_host_allocator.free(ptr);
#endif
}
// END of DECLARE_HOST_ALLOCATOR

REGISTER_HOST_ALLOCATOR(
    at::kPrivateUse1,
    &caching_host_allocator
)

bool ptr_exist(void* ptr)
{
    return caching_host_allocator.impl_->ptr_check(ptr);
}

c10::Allocator* getPinnedMemoryAllocator()
{
    innerInitNPU();
    return at::getHostAllocator(at::kPrivateUse1);
}

} // namespace at_npu::native