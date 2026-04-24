#pragma once

#include <unordered_map>
#include <map>
#include <memory>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMInterface.h"
#include "torch_npu/csrc/logging/LogContext.h"

namespace c10d {
namespace symmetric_memory {

inline std::shared_ptr<npu_logging::Logger>& GetSymmMemLogger()
{
    static std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger("torch_npu.symmetric_memory");
    return logger;
}

#define TORCH_NPU_SYMMEM_LOGD(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGD(c10d::symmetric_memory::GetSymmMemLogger(), format, ##__VA_ARGS__); \
        ASCEND_LOGD(format, ##__VA_ARGS__);                                    \
    } while (0);

struct NPUSHMEMAllocation {
    void* ptr;
    size_t buffer_size;
    int device_idx;

    NPUSHMEMAllocation(void* ptr, size_t buffer_size, int device_idx)
        : ptr(ptr), buffer_size(buffer_size), device_idx(device_idx) {}

    ~NPUSHMEMAllocation();
};

class NPUSHMEMSymmetricMemory : public SymmetricMemory {
public:
    NPUSHMEMSymmetricMemory(
        std::shared_ptr<NPUSHMEMAllocation> allocation,
        const std::string& group_name);

    ~NPUSHMEMSymmetricMemory() override;

    std::vector<void*> get_buffer_ptrs() override;
    std::vector<void*> get_signal_pad_ptrs() override;
    void** get_buffer_ptrs_dev() override;
    void** get_signal_pad_ptrs_dev() override;
    size_t get_buffer_size() override;
    size_t get_offset() override;

    bool has_multicast_support() override;
    void* get_multicast_ptr() override;

    void barrier(int channel, size_t timeout_ms) override;
    void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
    void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

    int get_rank() override;
    int get_world_size() override;
    c10::Device get_device() override;

    const std::vector<int>& get_rank_to_global_rank() override;

private:
    std::shared_ptr<NPUSHMEMAllocation> allocation_;
    size_t buffer_size_;
    std::vector<void*> buffers_;
    std::vector<void*> signal_pads_;
    int device_idx_;
    int rank_;
    int world_size_;
    void** buffers_dev_;
    void** signal_pads_dev_;
    std::string group_name_;
};

class NPUSHMEMSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
public:
    void* alloc(
        size_t size,
        int device_idx,
        const std::optional<std::string>& group_name) override;

    void free(void* ptr) override;

    size_t get_alloc_size(void* ptr) override;

    c10::intrusive_ptr<SymmetricMemory> rendezvous(
        void* ptr,
        const std::optional<std::string>& group_name) override;

    bool has_multicast_support(int device_idx) override;

    c10::DeviceType supported_device_type() override;

    std::string name() override;

private:
    std::unordered_map<void*, std::shared_ptr<NPUSHMEMAllocation>> allocations_;
    std::map<std::tuple<void*, std::string>, c10::intrusive_ptr<SymmetricMemory>>
        symm_mems_;
};

} // namespace symmetric_memory
} // namespace c10d
