#pragma once
#include <c10/core/Allocator.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace torch_npu {
namespace ipc {

TORCH_NPU_API bool NpuIPCCollect();

struct NpuIPCReceivedData final {
    NpuIPCReceivedData() = default;
    explicit NpuIPCReceivedData(std::shared_ptr<void> shared_ptr)
        : shared_ptr_(std::move(shared_ptr)) {}
    std::shared_ptr<void> shared_ptr_;
};

struct NpuIPCSentData final {
    std::string handle_;
    uint64_t offset_;
    uint64_t* counter_ptr_;     // Reference counter shared memory block
    at::DataPtr original_ptr_;  // Original mem allocation
    char* event_;               // Sync event
    bool event_sync_required_;
    at::Device device_;

    NpuIPCSentData(
        std::string handle,
        uint64_t offset,
        uint64_t* counter_ptr,
        at::Device device);
    ~NpuIPCSentData();

    uint64_t counter_value();
    std::string handle()
    {
        return handle_;
    }
    uint64_t offset()
    {
        return offset_;
    }
    void set_original_ptr(at::DataPtr data_ptr)
    {
        original_ptr_ = std::move(data_ptr);
    }
};

TORCH_NPU_API at::DataPtr GetNewRefCountedSentData(
    void* data,
    at::Device device);

namespace {

inline constexpr int64_t NPU_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t NPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;
inline constexpr int64_t NPU_IPC_MAXIMUM_EVENTS_TO_USE = 0;

// All to be deleted data blocks with non zero reference counter goes there
struct NpuIPCSentDataLimbo final {
    ~NpuIPCSentDataLimbo();
    bool collect();
    void add(std::unique_ptr<NpuIPCSentData> shared_block);
    uint64_t size();

private:
    std::vector<std::unique_ptr<NpuIPCSentData>> shared_blocks_;
    std::mutex limbo_mutex_;
};

struct NpuIPCRefCountersFile final {
    NpuIPCRefCountersFile(
        std::string handle,
        uint64_t size,
        at::DataPtr data_ptr)
        :   size_(size),
            handle_(std::move(handle)),
            refcounted_shared_mem_(std::move(data_ptr)) {}

    uint64_t* counter_ptr()
    {
        return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
    }

    void set_counter(uint64_t value)
    {
        *counter_ptr() = value;
    }

    bool have_offsets()
    {
        return next_offset_ < size_;
    }

    bool offsets_in_use()
    {
        return used_slots_;
    }

    uint64_t get_offset()
    {
        return next_offset_;
    }

    void rotate_offset()
    {
        next_offset_++;
        used_slots_++;
    }

    void return_offset(uint64_t offset /* unused */)
    {
        used_slots_--;
    }

    std::string handle()
    {
        return handle_;
    }

private:
    uint64_t next_offset_{0};
    uint64_t size_;
    uint64_t used_slots_{0};
    std::string handle_;
    at::DataPtr refcounted_shared_mem_;
};

} // namespace
} // namespace ipc
} // namespace torch_npu

namespace c10_npu {
namespace NPUCachingAllocator {
namespace {

class NpuIPCCollectCallback : public FreeMemoryCallback {
public:
    bool Execute() override
    {
        return torch_npu::ipc::NpuIPCCollect();
    }
};

} // namespace
} // namespace NPUCachingAllocator
} // namespace c10_npu