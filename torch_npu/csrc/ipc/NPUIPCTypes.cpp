#include <ATen/MapAllocator.h>
#include <atomic>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/ipc/NPUIPCTypes.h"

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace torch_npu {
namespace ipc {

namespace {

void warnProducerTerminatedBeforeSharedTensorsReleased()
{
    static bool warned = false;
    if (!warned) {
        LOG(WARNING)
            << "Producer process has been terminated before all shared NPU tensors released. See Note [Sharing NPU tensors]";
        warned = true;
    }
}

struct NpuIPCGlobalEntities {
    // This class is used as a singleton (see npu_ipc_global_entities)
    // This variable is used to track its lifetime to avoid accessing it
    // after it was destroyed which would lead to segmentation faults
    // Note that a trvial type is used which doesn't suffer from construction
    // and destruction order issues
    static bool alive;

    std::mutex ref_counters_mutex_;
    std::atomic<int64_t> sync_events_used_{0};
    std::map<std::string, std::shared_ptr<NpuIPCRefCountersFile>>
        ref_counters_files_;
    std::shared_ptr<NpuIPCRefCountersFile> next_available_ref_counters_file_;
    NpuIPCSentDataLimbo NpuIPCSentDataLimbo_;

    NpuIPCGlobalEntities()
    {
        alive = true;
    }

    ~NpuIPCGlobalEntities()
    {
        NpuIPCSentDataLimbo_.collect();
        safe_clean_current_file();
        if (next_available_ref_counters_file_) {
            warnProducerTerminatedBeforeSharedTensorsReleased();
        }
        alive = false;
    }

    void safe_clean_current_file()
    {
        std::lock_guard<std::mutex> lock(ref_counters_mutex_);
        if (next_available_ref_counters_file_ &&
            next_available_ref_counters_file_->offsets_in_use() == 0) {
            ref_counters_files_.erase(next_available_ref_counters_file_->handle());
            next_available_ref_counters_file_.reset();
        }
    }
};

bool NpuIPCGlobalEntities::alive = false;
NpuIPCGlobalEntities npu_ipc_global_entities;

NpuIPCSentDataLimbo::~NpuIPCSentDataLimbo()
{
    collect();
    if (size() > 0) {
        warnProducerTerminatedBeforeSharedTensorsReleased();
    }
}

bool NpuIPCSentDataLimbo::collect()
{
    bool freed_memory = false;
    std::vector<std::unique_ptr<NpuIPCSentData>> reset_blocks;
    {
        // Begin critical section to modify shared blocks
        std::lock_guard<std::mutex> lock(limbo_mutex_);
        std::vector<std::unique_ptr<NpuIPCSentData>> kept_blocks;
        for (auto& sd : shared_blocks_) {
            if (sd->counter_value() > 0) {
                kept_blocks.push_back(std::move(sd));
            } else {
                freed_memory = true;
                reset_blocks.push_back(std::move(sd));
            }
        }
        shared_blocks_ = std::move(kept_blocks);
    }
    // Need to reset blocks out of the critical section here, otherwise it
    // deadlocks.
    for (auto& sd : reset_blocks) {
        sd.reset();
    }
    return freed_memory;
}

void NpuIPCSentDataLimbo::add(std::unique_ptr<NpuIPCSentData> shared_block)
{
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    static bool warned = false;
    if (shared_blocks_.size() > NPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO &&
        !warned) {
        LOG(WARNING)
            << "Producer process tried to deallocate over "
            << NPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
            << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down. "
            << "We assume it will never going to be the case.";
        warned = true;
    }
    shared_blocks_.push_back(std::move(shared_block));
}

uint64_t NpuIPCSentDataLimbo::size()
{
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    return shared_blocks_.size();
}

void NpuIPCSentDataDelete(void* ptr)
{
    std::unique_ptr<NpuIPCSentData> sent_data(
        static_cast<NpuIPCSentData*>(ptr));
    if (!NpuIPCGlobalEntities::alive) {
        return;
    }
    if (sent_data->counter_value() > 0) {
        npu_ipc_global_entities.NpuIPCSentDataLimbo_.add(std::move(sent_data));
    }
    npu_ipc_global_entities.NpuIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */)
{
    if (!NpuIPCGlobalEntities::alive) {
        return;
    }
    std::lock_guard<std::mutex> lock(
        npu_ipc_global_entities.ref_counters_mutex_);
    auto& map = npu_ipc_global_entities.ref_counters_files_;
    auto it = map.find(handle);
    if (it != map.end()) {
        it->second->return_offset(offset);
        if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
            map.erase(handle);
        }
    }
}

} // namespace

NpuIPCSentData::NpuIPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    :   handle_(std::move(handle)),
        offset_(offset),
        counter_ptr_(counter_ptr),
        device_(device)
{
    if (npu_ipc_global_entities.sync_events_used_.load() <
        NPU_IPC_MAXIMUM_EVENTS_TO_USE) {
        // NPU does not suppurt event_sync in IPC now.
    } else {
        auto stream = c10_npu::getCurrentNPUStream(device.index());
        c10_npu::stream_synchronize(stream);
        event_ = nullptr;
        event_sync_required_ = false;
    }
}

NpuIPCSentData::~NpuIPCSentData()
{
    ReturnRefCounter(handle_, offset_);
    try {
        if (event_sync_required_) {
            // NPU does not suppurt event_sync in IPC now.
        }
    } catch (...) { /* No throw */
    }
}

uint64_t NpuIPCSentData::counter_value()
{
    return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device)
{
    {
        std::lock_guard<std::mutex> lock(
            npu_ipc_global_entities.ref_counters_mutex_);
        if (!npu_ipc_global_entities.next_available_ref_counters_file_) {
            std::string ref_counter_handle = at::NewProcessWideShmHandle();

            int flags =
                at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
            at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
                ref_counter_handle.c_str(),
                flags,
                sizeof(int64_t) * NPU_IPC_REF_COUNTER_FILE_SIZE,
                nullptr);
            auto rc = std::make_shared<NpuIPCRefCountersFile>(
                ref_counter_handle, NPU_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
            npu_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
            npu_ipc_global_entities.next_available_ref_counters_file_ = rc;
        }
    }
    npu_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
    auto sent_data = new NpuIPCSentData(
        npu_ipc_global_entities.next_available_ref_counters_file_->handle(),
        npu_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
        npu_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
        device);

    npu_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
    if (!npu_ipc_global_entities.next_available_ref_counters_file_
            ->have_offsets()) {
        npu_ipc_global_entities.next_available_ref_counters_file_.reset();
    }
    return at::DataPtr(data, sent_data, NpuIPCSentDataDelete, device);
}

bool NpuIPCCollect()
{
    if (!NpuIPCGlobalEntities::alive) {
        return true;
    }
    bool freed_memory = npu_ipc_global_entities.NpuIPCSentDataLimbo_.collect();
    if (npu_ipc_global_entities.NpuIPCSentDataLimbo_.size() == 0) {
        npu_ipc_global_entities.safe_clean_current_file();
    }
    return freed_memory;
}

} // namespace ipc
} // namespace torch_npu

namespace c10_npu {
namespace NPUCachingAllocator {

REGISTER_FREE_MEMORY_CALLBACK("npu_ipc_collect", NpuIPCCollectCallback);

} // namespace NPUCachingAllocator
} // namespace c10_npu