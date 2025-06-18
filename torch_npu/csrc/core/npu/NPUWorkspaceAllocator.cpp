#include <memory>
#include <vector>

#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"

#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/profiler/npu_profiler.h"
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

namespace at_npu {
namespace native {

at::Tensor allocate_workspace(uint64_t workspace_size, aclrtStream stream)
{
    return at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, stream);
}

} // namespace native
} // namespace at_npu


namespace c10_npu {
namespace NPUWorkspaceAllocator {

namespace {
constexpr size_t kRoundLarge = 2097152; // Alloceted memory is aligned to 2 MiB.
} // namespace

struct WorkspaceBlock {
    void* data_ptr;
    size_t size;
    bool allocated = 0;
    int64_t requested_size = 0;
    std::shared_ptr<c10::GatheredContext> context_when_allocated = nullptr;
    WorkspaceBlock() : data_ptr(nullptr), size(0) {}
};

void update_stat(Stat &stat, int64_t amount)
{
    stat.current += amount;
    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0) {
        stat.allocated += amount;
    }
    if (amount < 0) {
        stat.freed += -amount;
    }
}

class DeviceWorkspaceAllocator {
public:
    DeviceWorkspaceAllocator()
    {
        blocks.clear();
        context_recorder_.store(nullptr);
    }

    std::shared_ptr<c10::GatheredContext> maybeGatherContext(RecordContext level)
    {
        if (record_context_ < level) {
            return nullptr;
        }
        return context_recorder_.load()();
    }
    
    void* malloc(size_t size, aclrtStream stream)
    {
        auto context = maybeGatherContext(RecordContext::STATE);

        size_t alloc_size = size + 32;

        auto it = blocks.find(stream);
        if (it == blocks.end()) {
            blocks.emplace(stream, new WorkspaceBlock());
        }

        WorkspaceBlock* block = blocks[stream];
        if (block->size < alloc_size) {
            if (block->data_ptr != nullptr) {
                ASCEND_LOGI("NPUWorkspaceAllocator free by aclrtFree: size=%zu", block->size);
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeDeviceWithTimeout());
                NPU_CHECK_ERROR(aclrtFree(block->data_ptr));
                update_stat(stats.reserved_bytes, -block->size);
#ifndef BUILD_LIBTORCH
                if (torch_npu::profiler::MstxMgr::GetInstance()->isMsleaksEnable()) {
                    mstxDomainHandle_t msleaksDomain = torch_npu::profiler::MstxMgr::GetInstance()->createLeaksDomain(torch_npu::profiler::DOMAIN_MSLEAKS.c_str());
                    torch_npu::profiler::MstxMgr::GetInstance()->memRegionsUnregister(msleaksDomain, block->data_ptr);
                }
                record_mem_size_decrement(block->size);
                const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
                if (C10_UNLIKELY(trigger)) {
                    trigger->traceNpuMemoryDeallocation(
                        reinterpret_cast<uintptr_t>(block->data_ptr));
                }
                torch_npu::profiler::reportMemoryDataToNpuProfiler({
                    static_cast<int8_t>(c10::DeviceType::PrivateUse1),
                    device,
                    static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::WORKSPACE_ALLOCATOR),
                    static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_FREE),
                    static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_INNER),
                    reinterpret_cast<int64_t>(block->data_ptr),
                    -block->size,
                    stats.allocated_bytes.current,
                    stats.reserved_bytes.current,
                    stats.allocated_bytes.current,
                    reinterpret_cast<int64_t>(stream)}
                );
#endif
                block->data_ptr = nullptr;
            }

            block->size = kRoundLarge * ((alloc_size + kRoundLarge - 1) / kRoundLarge);

            TORCH_CHECK(
                alloc_size <= block->size,
                "The allocated memory ", block->size, " bytes is smaller than the required memory ", alloc_size, " bytes.",
                PTA_ERROR(ErrCode::MEMORY));

            aclError err = c10_npu::acl::AclrtMallocAlign32(
                &block->data_ptr, block->size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_ONLY);
            if (err != ACL_ERROR_NONE) {
                return nullptr;
            }
            block->context_when_allocated = std::move(context);
            block->requested_size = size;

            ASCEND_LOGD("NPUWorkspaceAllocator malloc by AclrtMallocAlign32: size=%zu", block->size);
            update_stat(stats.reserved_bytes, block->size);
#ifndef BUILD_LIBTORCH
            if (torch_npu::profiler::MstxMgr::GetInstance()->isMsleaksEnable()) {
                mstxDomainHandle_t msleaksDomain = torch_npu::profiler::MstxMgr::GetInstance()->createLeaksDomain(torch_npu::profiler::DOMAIN_MSLEAKS.c_str());
                mstxMemVirtualRangeDesc_t desc{device, block->data_ptr, block->size};
                torch_npu::profiler::MstxMgr::GetInstance()->memRegionsRegister(msleaksDomain, &desc);
            }
            record_mem_size_increment(block->size);
            torch_npu::profiler::reportMemoryDataToNpuProfiler({
                static_cast<int8_t>(c10::DeviceType::PrivateUse1),
                device,
                static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::WORKSPACE_ALLOCATOR),
                static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_MALLOC),
                static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_INNER),
                reinterpret_cast<int64_t>(block->data_ptr),
                block->size,
                stats.allocated_bytes.current,
                stats.reserved_bytes.current,
                stats.allocated_bytes.current,
                reinterpret_cast<int64_t>(stream)}
            );
            this->last_block = block;
            this->last_stream = stream;
            const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
            if (C10_UNLIKELY(trigger)) {
                trigger->traceNpuMemoryAllocation(
                    reinterpret_cast<uintptr_t>(block->data_ptr));
            }
#endif
        }

        allocated_size = block->size;
        update_stat(stats.allocated_bytes, block->size);
#ifndef BUILD_LIBTORCH
        torch_npu::profiler::reportMemoryDataToNpuProfiler({
            static_cast<int8_t>(c10::DeviceType::PrivateUse1),
            device,
            static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::WORKSPACE_ALLOCATOR),
            static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_MALLOC),
            static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_INNER),
            reinterpret_cast<int64_t>(block->data_ptr),
            block->size,
            stats.allocated_bytes.current,
            stats.reserved_bytes.current,
            stats.allocated_bytes.current,
            reinterpret_cast<int64_t>(stream)}
        );
        this->last_block = block;
        this->last_stream = stream;
#endif
        return block->data_ptr;
    }

    void free()
    {
        update_stat(stats.allocated_bytes, -allocated_size);
#ifndef BUILD_LIBTORCH
        if (this->last_block && this->last_block->data_ptr && this->last_stream) {
            torch_npu::profiler::reportMemoryDataToNpuProfiler({
                static_cast<int8_t>(c10::DeviceType::PrivateUse1),
                device,
                static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::WORKSPACE_ALLOCATOR),
                static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_FREE),
                static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_INNER),
                reinterpret_cast<int64_t>(this->last_block->data_ptr),
                -allocated_size,
                stats.allocated_bytes.current,
                stats.reserved_bytes.current,
                stats.allocated_bytes.current,
                reinterpret_cast<int64_t>(this->last_stream)}
            );
        }
#endif
    }

    // return to the system allocator
    void empty_cache(bool need_empty_queue, bool check_error)
    {
        if (need_empty_queue) {
            ASCEND_LOGI("NPUWorkspaceAllocator empty_cache in main_thread.");
            c10_npu::emptyAllNPUStream(check_error);
        } else {
            ASCEND_LOGI("NPUWorkspaceAllocator empty_cache in acl_thread.");
        }

        auto acl_ret = c10_npu::acl::AclrtSynchronizeDeviceWithTimeout();
        if (check_error) {
            NPU_CHECK_ERROR(acl_ret, "AclrtSynchronizeDeviceWithTimeout");
        } else {
            NPU_CHECK_WARN(acl_ret);
        }

        for (const auto& block_pair : blocks) {
            if (block_pair.second->data_ptr != nullptr) {
                ASCEND_LOGI("NPUWorkspaceAllocator free by aclrtFree: size=%zu", block_pair.second->size);
                NPU_CHECK_ERROR(aclrtFree(block_pair.second->data_ptr));
                update_stat(stats.reserved_bytes, -block_pair.second->size);
#ifndef BUILD_LIBTORCH
                if (torch_npu::profiler::MstxMgr::GetInstance()->isMsleaksEnable()) {
                    mstxDomainHandle_t msleaksDomain = torch_npu::profiler::MstxMgr::GetInstance()->createLeaksDomain(torch_npu::profiler::DOMAIN_MSLEAKS.c_str());
                    torch_npu::profiler::MstxMgr::GetInstance()->memRegionsUnregister(msleaksDomain, block_pair.second->data_ptr);
                }
                record_mem_size_decrement(block_pair.second->size);
                const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
                if (C10_UNLIKELY(trigger)) {
                    trigger->traceNpuMemoryDeallocation(
                        reinterpret_cast<uintptr_t>(block_pair.second->data_ptr));
                }
                torch_npu::profiler::reportMemoryDataToNpuProfiler({
                    static_cast<int8_t>(c10::DeviceType::PrivateUse1),
                    device,
                    static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::WORKSPACE_ALLOCATOR),
                    static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_FREE),
                    static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_INNER),
                    reinterpret_cast<int64_t>(block_pair.second->data_ptr),
                    -block_pair.second->size,
                    stats.allocated_bytes.current,
                    stats.reserved_bytes.current,
                    stats.allocated_bytes.current,
                    reinterpret_cast<int64_t>(block_pair.first)}
                );
#endif
            }
            delete block_pair.second;
        }

        blocks.clear();
    }

    void record_history(bool enabled, CreateContextFn context_recorder, RecordContext when)
    {
        TORCH_CHECK(when == RecordContext::NEVER || context_recorder, PTA_ERROR(ErrCode::INTERNAL));
        record_flag = enabled;
        context_recorder_.store(record_flag ? context_recorder : nullptr);
        record_context_ = enabled ? when : RecordContext::NEVER;
    }

    std::vector<TraceEntry> get_trace()
    {
        std::vector<TraceEntry> alloc_trace;
#ifndef BUILD_LIBTORCH
        if (!record_flag) {
            return alloc_trace;
        }
        for (const auto& block_pair : blocks) {
            auto te = TraceEntry(TraceEntry::WORKSPACE_SNAPSHOT, device, int64_t(block_pair.second->data_ptr),
                                 block_pair.second->size, block_pair.first,
                                 record_context_ >= RecordContext::ALLOC ? block_pair.second->context_when_allocated
                                                                         : nullptr);
            alloc_trace.emplace_back(te);

            te = TraceEntry(TraceEntry::SEGMENT_ALLOC, device, int64_t(block_pair.second->data_ptr),
                            block_pair.second->size, block_pair.first,
                            record_context_ >= RecordContext::ALLOC ? block_pair.second->context_when_allocated
                                                                    : nullptr);
            alloc_trace.emplace_back(te);

            te = TraceEntry(TraceEntry::ALLOC, device, int64_t(block_pair.second->data_ptr), block_pair.second->size,
                            block_pair.first,
                            record_context_ >= RecordContext::ALLOC ? block_pair.second->context_when_allocated
                                                                    : nullptr);
            alloc_trace.emplace_back(te);
        }
#endif
        return alloc_trace;
    }

    std::vector<SegmentInfo> get_segm()
    {
        std::vector<SegmentInfo> result;
#ifndef BUILD_LIBTORCH
        for (const auto& block_pair : blocks) {
            result.emplace_back();
            SegmentInfo& segment_info = result.back();
            segment_info.device = device;
            segment_info.address = reinterpret_cast<int64_t>(block_pair.second->data_ptr);
            segment_info.stream = block_pair.first;
            segment_info.is_large = true;
            segment_info.is_expandable = false;
            segment_info.context_when_allocated = block_pair.second->context_when_allocated;

            const WorkspaceBlock* block = block_pair.second;
            segment_info.blocks.emplace_back();
            BlockInfo& block_info = segment_info.blocks.back();
            block_info.size = block->size;
            block_info.requested_size = block->requested_size;
            block_info.allocated = block->allocated;
            block_info.active = block->allocated;

            segment_info.total_size += block_info.size;
            if (block_info.allocated) {
                segment_info.allocated_size += block_info.size;
                segment_info.active_size += block_info.size;
                segment_info.requested_size += block_info.requested_size;
            }
            block_info.context_when_allocated = block->context_when_allocated;
        }
#endif
        return result;
    }

#ifndef BUILD_LIBTORCH
    void set_device(int device_id)
    {
        this->device = device_id;
    }

    void record_mem_size_increment(size_t size)
    {
        this->sum_mem += size;
    }

    void record_mem_size_decrement(size_t size)
    {
        this->sum_mem -= size;
    }

    uint64_t get_mem_size()
    {
        return sum_mem;
    }
#endif

    DeviceStats getStats()
    {
        return stats;
    }

    void *getStreamPtr(aclrtStream stream)
    {
        auto it = blocks.find(stream);
        if (it == blocks.end()) {
            return nullptr;
        }
        WorkspaceBlock *block = it->second;
        return block->data_ptr;
    }

private:
    ska::flat_hash_map<aclrtStream, WorkspaceBlock*> blocks;
    bool record_flag = false;
    std::atomic<CreateContextFn> context_recorder_;
    RecordContext record_context_ = RecordContext::NEVER;
#ifndef BUILD_LIBTORCH
    uint64_t sum_mem = 0;
    int device = 0;
    aclrtStream last_stream = nullptr;
    WorkspaceBlock* last_block = nullptr;
#endif
    DeviceStats stats;
    size_t allocated_size = 0;
}; // class DeviceworkspaceAllocator

static void uncached_delete(void* ptr)
{
    NPU_CHECK_WARN(c10_npu::acl::AclrtSynchronizeDeviceWithTimeout());
    NPU_CHECK_ERROR(aclrtFree(ptr));
}

static void local_raw_delete(void* ptr);

class NpuWorkspaceAllocator : public c10::Allocator {
private:
    // allocated blocks by device pointer
    ska::flat_hash_map<void *, int> allocated_ptrs;

    void replace_allocated_ptr(void *new_ptr, void *src_ptr, int device)
    {
        auto it = allocated_ptrs.find(src_ptr);
        if (it != allocated_ptrs.end()) {
            allocated_ptrs.erase(it);
        }
        allocated_ptrs[new_ptr] = device;
    }

    int get_allocated_device(void *ptr)
    {
        auto it = allocated_ptrs.find(ptr);
        if (it == allocated_ptrs.end()) {
            return -1;
        }
        return it->second;
    }
public:
    std::vector<std::unique_ptr<DeviceWorkspaceAllocator>> device_allocator;

    void init(int device_count)
    {
        int size = static_cast<int>(device_allocator.size());
        if (size < device_count) {
            device_allocator.resize(device_count);
            for (const auto i : c10::irange(size, device_count)) {
                device_allocator[i] = std::make_unique<DeviceWorkspaceAllocator>();
#ifndef BUILD_LIBTORCH
                device_allocator[i]->set_device(i);
#endif
            }
        }
    }

    void malloc(void** new_ptr, int device, size_t size, aclrtStream stream)
    {
        auto src_ptr = static_cast<void*>(device_allocator[device]->getStreamPtr(stream));
        *new_ptr = static_cast<void*>(device_allocator[device]->malloc(size, stream));

        // Free all cached blocks and try again.
        if ((*new_ptr) == nullptr) {
            device_allocator[device]->empty_cache(false, true);
            c10_npu::NPUCachingAllocator::emptyCache(true);
            *new_ptr = static_cast<void*>(device_allocator[device]->malloc(size, stream));
        }

        if ((*new_ptr) == nullptr) {
            size_t device_free;
            size_t device_total;
            NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));

            TORCH_CHECK(false,
                "NPU out of memory. NPUWorkspaceAllocator tried to allocate ",
                format_size(size),
                "(NPU ", device, "; ",
                format_size(device_total),
                " total capacity; ",
                format_size(device_free),
                " free)",
                PTA_ERROR(ErrCode::MEMORY));
        }

        if ((*new_ptr) != src_ptr) {
            replace_allocated_ptr(*new_ptr, src_ptr, device);
        }
    }

    void empty_cache(int device, bool need_empty_queue, bool check_error)
    {
        device_allocator[device]->empty_cache(need_empty_queue, check_error);
        allocated_ptrs.clear();
    }

    void record_history(bool enabled, CreateContextFn context_recorder, RecordContext when)
    {
        for (auto& allocator : device_allocator) {
            allocator->record_history(enabled, context_recorder, when);
        }
    }

    SnapshotInfo snapshot()
    {
        SnapshotInfo result;
        int count = static_cast<int>(device_allocator.size());
        for (int i = 0; i < count; i++) {
            result.device_traces.emplace_back(device_allocator[i]->get_trace());
            auto snap = device_allocator[i]->get_segm();
            result.segments.insert(result.segments.end(), snap.begin(), snap.end());
        }
        return result;
    }

    c10::DataPtr allocate(size_t size) override
    {
        int device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        void* dev_ptr = nullptr;
        void (*delete_func)(void*) = &local_raw_delete;
        return {dev_ptr, dev_ptr, delete_func, c10::Device(c10::DeviceType::PrivateUse1, device)};
    }

    c10::DataPtr allocate_with_stream(size_t size, aclrtStream stream)
    {
        int device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        void* dev_ptr = nullptr;
        void (*delete_func)(void*) = &local_raw_delete;

        if (c10_npu::option::OptionsManager::CheckForceUncached()) {
            delete_func = &uncached_delete;
            if (size != 0) {
                size_t alloc_size = size + 32;
                NPU_CHECK_ERROR(
                    c10_npu::acl::AclrtMallocAlign32(&dev_ptr, alloc_size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_ONLY));
            }
        } else {
            if (size != 0) {
                this->malloc(&dev_ptr, device, size, stream);
            }
        }
        return {dev_ptr, dev_ptr, delete_func, c10::Device(c10::DeviceType::PrivateUse1, device)};
    }

    c10::DeleterFnPtr raw_deleter() const override
    {
        if (c10_npu::option::OptionsManager::CheckForceUncached()) {
            return &uncached_delete;
        } else {
            return &local_raw_delete;
        }
    }

    // Note [COW/lazy_clone is not supported yet]
    void copy_data(void* dest, const void* src, std::size_t count) const final
    {
        default_copy_data(dest, src, count);
    }

    void assertValidDevice(int device)
    {
        const auto device_num = device_allocator.size();
        TORCH_CHECK(0 <= device && device < static_cast<int64_t>(device_num), "Invalid device argument ", device,
                    ": did you call init?", PTA_ERROR(ErrCode::PARAM));
    }

    void free(void* ptr)
    {
        if (!ptr) {
            return;
        }
        int device = get_allocated_device(ptr);
        if (device != -1) {
            device_allocator[device]->free();
        }
    }

    DeviceStats getDeviceStats(int device)
    {
        assertValidDevice(device);
        return device_allocator[device]->getStats();
    }
}; // class NpuWorkspaceAllocator

NpuWorkspaceAllocator workspace_allocator;

// Now we will reuse the allocated memory and not release immediately until
// memory is insufficient for NpuCachingAllocator or NpuWorkspaceAllocator.
// Then both will empty cache and the large memory will be released.
static void local_raw_delete(void* ptr)
{
    workspace_allocator.free(ptr);
}

c10::Allocator* get()
{
    return &workspace_allocator;
}

void init()
{
    uint32_t device_count = 0;
    NPU_CHECK_ERROR(aclrtGetDeviceCount(&device_count));
    workspace_allocator.init(device_count);
}

c10::DataPtr malloc_with_stream(size_t size, aclrtStream stream)
{
    return workspace_allocator.allocate_with_stream(size, stream);
}

void emptyCache(int device, bool need_empty_queue, bool check_error)
{
    workspace_allocator.empty_cache(device, need_empty_queue, check_error);
}

void recordHistory(bool enabled, CreateContextFn context_recorder, RecordContext when)
{
    workspace_allocator.record_history(enabled, context_recorder, when);
}
SnapshotInfo snapshot()
{
    return workspace_allocator.snapshot();
}

DeviceStats getDeviceStats(int device)
{
    return workspace_allocator.getDeviceStats(device);
}


} // namespace NPUWorkspaceAllocator
} // namespace c10_npu
