#include <c10/core/DeviceGuard.h>
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

namespace at_npu {
namespace native {

namespace {
struct BlockSize {
    size_t size; // allocation size
    void *ptr;   // host memory pointer

    explicit BlockSize(size_t size, void *ptr = nullptr) : size(size), ptr(ptr) {}
};

struct Block : public BlockSize {
    bool allocated;  // true if the block is currently allocated
    int event_count; // number of outstanding npu events
    std::unordered_set<c10_npu::NPUStream> streams;
    Block(size_t size, void *ptr, bool allocated)
        : BlockSize(size, ptr), allocated(allocated), event_count(0), streams() {}
};

class EventPool {
public:
    using Event = std::unique_ptr<
        c10_npu::NPUEvent,
        std::function<void(c10_npu::NPUEvent *)>>;
    EventPool() : pools_(c10_npu::device_count()) {}

    Event get(at::DeviceIndex device)
    {
        TORCH_INTERNAL_ASSERT(0 <= device, PTA_ERROR(ErrCode::PARAM));
        TORCH_INTERNAL_ASSERT(device < static_cast<at::DeviceIndex>(pools_.size()), PTA_ERROR(ErrCode::PARAM));
        auto &pool = pools_[device];
        auto destructor = [&pool](c10_npu::NPUEvent *event) {
            std::lock_guard<std::mutex> g(pool.mutex_);
            pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
        };

        // Try to acquire an event from the per-device pool.
        {
            std::lock_guard<std::mutex> g(pool.mutex_);
            if (!pool.event_pool_.empty()) {
                auto *event = pool.event_pool_.back().release();
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
        for (auto &pool : pools_) {
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

static bool BlockComparator(const BlockSize &a, const BlockSize &b)
{
    // sort by size, break ties with pointer
    if (a.size != b.size) {
        return a.size < b.size;
    }
    return reinterpret_cast<uintptr_t>(a.ptr) < reinterpret_cast<uintptr_t>(b.ptr);
}

struct HostAllocator {
    using Comparison = bool (*)(const BlockSize &, const BlockSize &);

    HostAllocator() : available(BlockComparator) {}

    aclError malloc(void **ptr, size_t size)
    {
        std::lock_guard<std::mutex> lock(mutex);

        // process outstanding npu events which may have occurred
        aclError err = processEvents();
        if (err != ACL_ERROR_NONE) {
            return err;
        }

        // search for the smallest block which can hold this allocation
        BlockSize search_key(size);
        auto it = available.lower_bound(search_key);
        if (it != available.end()) {
            Block &block = blocks.at(it->ptr);
            AT_ASSERT(!block.allocated && block.event_count == 0, PTA_ERROR(ErrCode::PARAM));
            block.allocated = true;
            *ptr = block.ptr;
            available.erase(it);
            return ACL_ERROR_NONE;
        }

        *ptr = nullptr;
        // for pin_memory in dataloader, it should be set device first when new a thread
        if (c10_npu::GetLocalDevice() < 0) {
            c10_npu::SetCurrentDevice();
        }

        // allocate a new block if no cached allocation is found
        err = aclrtMallocHost(ptr, size);
        if (err != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
            return err;
        }

        blocks.insert({*ptr, Block(size, *ptr, true)});
        return ACL_ERROR_NONE;
    }

    aclError free(void *ptr)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ptr) {
            return ACL_ERROR_NONE;
        }

        auto it = blocks.find(ptr);
        AT_ASSERT(it != blocks.end(), PTA_ERROR(ErrCode::VALUE));

        Block &block = it->second;
        AT_ASSERT(block.allocated, PTA_ERROR(ErrCode::VALUE));

        // free (on valid memory) shouldn't fail, so mark unallocated before
        // we process the streams.
        block.allocated = false;

        // insert npu events for each stream on which this block was used. This
        aclError err = insertEvents(block);
        if (err != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err);
            return err;
        }

        if (block.event_count == 0) {
            // the block can be re-used if there are no outstanding npu events
            available.insert(block);
        }
        return ACL_ERROR_NONE;
    }

    aclError recordEvent(void *ptr, c10_npu::NPUStream stream)
    {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = blocks.find(ptr);
        if (it == blocks.end()) {
            // Sync when host memory is allocated by malloc
            aclError error = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream);
            if (error != ACL_ERROR_NONE) {
                CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(error);
                C10_NPU_SHOW_ERR_MSG();
                AT_ERROR("ACL stream synchronize failed.");
                return error;
            }
            return ACL_ERROR_NONE;
        }

        Block &block = it->second;
        AT_ASSERT(block.allocated, PTA_ERROR(ErrCode::VALUE));

        block.streams.insert(stream);
        return ACL_ERROR_NONE;
    }

    bool isPinndPtr(void *ptr)
    {
        std::lock_guard<std::mutex> lock(mutex);
        return blocks.find(ptr) != blocks.end();
    }

    aclError processEvents()
    {
        // Process outstanding npuEvents. Events that are completed are removed
        // from the queue, and the 'event_count' for the corresponding allocation
        // is decremented. Stops at the first event which has not been completed.
        // Since events on different devices or streams may occur out of order,
        // the processing of some events may be delayed.
        while (!npu_events.empty()) {
            auto &e = npu_events.front();
            EventPool::Event event = std::move(e.first);
            if (!event->query()) {
                e.first = std::move(event);
                break;
            }

            Block &block = blocks.at(e.second);
            block.event_count--;
            if (block.event_count == 0 && !block.allocated) {
                available.insert(block);
            }
            npu_events.pop_front();
        }
        return ACL_ERROR_NONE;
    }

    void emptyCache()
    {
        std::lock_guard<std::mutex> lock(mutex);

        // process outstanding npu events which may have occurred
        processEvents();

        // Release cached events from the event pool.
        event_pool_.empty_cache();

        // clear list of available blocks
        available.clear();

        // free and erase non-allocated blocks
        for (auto it = blocks.begin(); it != blocks.end();) {
            Block &block = it->second;
            if (aclrtFreeHost(block.ptr) != ACL_ERROR_NONE) {
                ASCEND_LOGE("free host pin failed!");
            }
            if (!block.allocated) {
                it = blocks.erase(it);
            } else {
                block.streams.clear();
                ++it;
            }
        }
    }

    aclError insertEvents(Block &block)
    {
        aclError err = ACL_ERROR_NONE;

        int prev_device = 0;
        err = c10_npu::GetDevice(&prev_device);
        if (err != ACL_ERROR_NONE) {
            return err;
        }

        std::unordered_set<c10_npu::NPUStream> streams(std::move(block.streams));
        for (auto it = streams.begin(); it != streams.end(); ++it) {
            err = c10_npu::SetDevice(it->device_index());
            if (err != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                break;
            }

            EventPool::Event event = event_pool_.get(it->device_index());
            event->record(*it);
            ASCEND_LOGI("Event: record HostAllocator is successfully executed, event=%p", event.get());

            block.event_count++;
            npu_events.emplace_back(std::move(event), block.ptr);
        }

        c10_npu::SetDevice(prev_device);

        return err;
    }

private:
    EventPool event_pool_;

    // lock around all operations
    std::mutex mutex;

    // blocks by pointer
    std::unordered_map<void *, Block> blocks;

    // pointers that are ready to be allocated (event_count=0)
    std::set<BlockSize, Comparison> available;

    // outstanding ACL events
    std::deque<std::pair<EventPool::Event, void *>> npu_events;
};
} // namespace

static HostAllocator& getHostAllocator()
{
    // Construct allocator inside a function to prevent initialization when import
    static HostAllocator allocator;
    return allocator;
}

aclError CachingHostAllocator_recordEvent(
    void *ptr,
    c10_npu::NPUStream stream)
{
    return getHostAllocator().recordEvent(ptr, stream);
}

bool CachingHostAllocator_isPinned(void *ptr)
{
    return getHostAllocator().isPinndPtr(ptr);
}

void CachingHostAllocator_emptyCache()
{
    getHostAllocator().emptyCache();
}

static void CachingHostDeleter(void *ptr)
{
#ifndef BUILD_LIBTORCH
    // check the current thread have hold GIL Lock.
    if (PyGILState_Check()) {
        // the current thread should not hold GIL.
        Py_BEGIN_ALLOW_THREADS
        getHostAllocator().free(ptr);
        Py_END_ALLOW_THREADS
    } else {
        getHostAllocator().free(ptr);
    }
#else
    getHostAllocator().free(ptr);
#endif
}

struct CachingHostAllocator final : public at::Allocator {
    at::DataPtr allocate(size_t size) override
    {
        AT_ASSERT(size >= 0, PTA_ERROR(ErrCode::VALUE));
        void *ptr = nullptr;
        if (size > 0) {
            if (getHostAllocator().malloc(&ptr, size) != ACL_ERROR_NONE) {
                ASCEND_LOGE("allocate host pinned memory fail");
            }
        }
        return {ptr, ptr, &CachingHostDeleter, at::DeviceType::CPU};
    }
    at::DeleterFnPtr raw_deleter() const override
    {
        return &CachingHostDeleter;
    }
    // Note [COW/lazy_clone is not supported yet]
    void copy_data(void* dest, const void* src, std::size_t count) const
    {
        TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for THNPUCachingHostAllocator", PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
};

static CachingHostAllocator caching_host_allocator;
at::Allocator *getCachingHostAllocator()
{
    return &caching_host_allocator;
}

c10::Allocator *getPinnedMemoryAllocator()
{
    C10_LOG_API_USAGE_ONCE("aten.init.npu");
    c10_npu::NpuSysCtrl::SysStatus status =
        c10_npu::NpuSysCtrl::GetInstance().Initialize();
    if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
        ASCEND_LOGE("Npu init fail.");
    }
    return getCachingHostAllocator();
}

} // namespace native
} // namespace at_npu
