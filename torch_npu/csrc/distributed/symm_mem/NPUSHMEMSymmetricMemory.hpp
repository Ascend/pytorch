#pragma once

#include <unordered_map>
#include <map>
#include <memory>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMInterface.h"

namespace c10d {
namespace symmetric_memory {

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
    size_t get_signal_pad_size() override;

    bool has_multicast_support() override;
    void* get_multicast_ptr() override;

    at::Tensor get_buffer(
        int rank,
        c10::IntArrayRef sizes,
        c10::ScalarType dtype,
        int64_t storage_offset) override;

    at::Tensor get_signal_pad(
        int rank,
        c10::IntArrayRef sizes,
        std::optional<c10::ScalarType> dtype,
        int64_t storage_offset) override;

    void barrier(int channel, size_t timeout_ms) override;
    void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
    void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

    int get_rank() override;
    int get_world_size() override;

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

    std::vector<int> rank_to_global_rank_;
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

private:
    std::unordered_map<void*, std::shared_ptr<NPUSHMEMAllocation>> allocations_;
    std::map<std::tuple<void*, std::string>, c10::intrusive_ptr<SymmetricMemory>>
        symm_mems_;
};

} // namespace symmetric_memory
} // namespace c10d
