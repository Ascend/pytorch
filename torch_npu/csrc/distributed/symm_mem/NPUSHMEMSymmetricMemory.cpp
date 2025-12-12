#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSymmetricMemoryUtils.hpp"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMSymmetricMemory.hpp"

namespace c10d {
namespace symmetric_memory {

/* Start of NPUSHMEMSymmetricMemory implementation */

constexpr size_t npu_signal_pad_size = 2048;
static std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger("torch_npu.symmetric_memory");
static NPUStoreExchange storeExchange = NPUStoreExchange("NPUSHMEMSymmetricMemory");

NPUSHMEMAllocation::~NPUSHMEMAllocation()
{
    // Avoid calling NPU functions after driver shutting down
    if (is_finalizing()) {
        return;
    }
    auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx);
    at::DeviceGuard device_guard(device);
    logger->debug("~NPUSHMEMAllocation, start Shmem_free, ptr is %p.", ptr);
    Shmem_free(ptr);  // shmem_free has no return value
    logger->debug("~NPUSHMEMAllocation, end Shmem_free, ptr is %p.", ptr);
}

NPUSHMEMSymmetricMemory::NPUSHMEMSymmetricMemory(
    std::shared_ptr<NPUSHMEMAllocation> allocation,
    const std::string& group_name)
    : allocation_(allocation),
    buffer_size_(allocation->buffer_size),
    device_idx_(allocation->device_idx),
    group_name_(group_name)
{
    // For logging only
    static int exchanged_n_times = 0;
    auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx_);
    at::DeviceGuard device_guard(device);

    auto global_rank = get_group_info("0").rank;
    GroupInfo& group_info = get_group_info(group_name_);
    auto store = group_info.store;
    rank_ = group_info.rank;
    world_size_ = group_info.world_size;
    // Exchange rank to global rank mapping for this group.
    // If it is already available, skip the exchange.
    if (group_info.rank_to_global_rank.empty()) {
        group_info.rank_to_global_rank =
            storeExchange.all_gather(store, rank_, world_size_, global_rank);
        exchanged_n_times++;
        if (rank_ == 0) {
            std::stringstream ss;
            for (size_t i = 0; i < group_info.rank_to_global_rank.size(); ++i) {
                ss << group_info.rank_to_global_rank[i];
                if (i != group_info.rank_to_global_rank.size() - 1) {
                    ss << ", ";
                }
            }
            logger->debug("[rank %d] rank_to_global_rank: %s, group_name: %s, exchanged_n_times: %d.",
                rank_, (ss.str()).c_str(), group_name_.c_str(), exchanged_n_times);
        }
    }
    TORCH_INTERNAL_ASSERT(!group_info.rank_to_global_rank.empty());
    rank_to_global_rank_ = group_info.rank_to_global_rank;
    for (int r = 0; r < world_size_; ++r) {
        auto buffer = Shmem_ptr(allocation->ptr, rank_to_global_rank_[r]);
        TORCH_CHECK(buffer != nullptr, "shmem_ptr return nullptr with ptr ", allocation->ptr, DIST_ERROR(ErrCode::MEMORY));
        buffers_.push_back(buffer);
        logger->debug("[rank %d] NPUSHMEMSymmetricMemory shmem_ptr, r is %d, rank_to_global_rank is %d, ptr is %p, shmem_ptr is %p.",
            rank_, r, rank_to_global_rank_[r], allocation->ptr, buffer);
    }

    // to be done
    // signal_pads_ buffers_dev_ signal_pads_dev_ rank_to_global_rank_dev_
    logger->debug("NPUSHMEMSymmetricMemory created, buffer_size is %d, device_idx is %d, group_name is %s.",
        allocation->buffer_size, allocation->device_idx, group_name_.c_str());
}

NPUSHMEMSymmetricMemory::~NPUSHMEMSymmetricMemory()
{
    // to be done
    logger->debug("NPUSHMEMSymmetricMemory destroy, group_name is %s", group_name_.c_str());
}

std::vector<void*> NPUSHMEMSymmetricMemory::get_buffer_ptrs()
{
    return buffers_;
}

std::vector<void*> NPUSHMEMSymmetricMemory::get_signal_pad_ptrs()
{
    return signal_pads_;
}

void** NPUSHMEMSymmetricMemory::get_buffer_ptrs_dev()
{
    return buffers_dev_;
}

void** NPUSHMEMSymmetricMemory::get_signal_pad_ptrs_dev()
{
    return signal_pads_dev_;
}

size_t NPUSHMEMSymmetricMemory::get_buffer_size()
{
    return buffer_size_;
}

size_t NPUSHMEMSymmetricMemory::get_signal_pad_size()
{
    return npu_signal_pad_size;
}

bool NPUSHMEMSymmetricMemory::has_multicast_support()
{
    return false;
}

void* NPUSHMEMSymmetricMemory::get_multicast_ptr()
{
    return nullptr;
}

at::Tensor NPUSHMEMSymmetricMemory::get_buffer(
    int rank,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    int64_t storage_offset)
{
    // to be done
    throw std::runtime_error("NPUSHMEMSymmetricMemory does not support get_buffer" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

at::Tensor NPUSHMEMSymmetricMemory::get_signal_pad(
    int rank,
    c10::IntArrayRef sizes,
    std::optional<c10::ScalarType> dtype,
    int64_t storage_offset)
{
    // to be done
    throw std::runtime_error("NPUSHMEMSymmetricMemory does not support get_signal_pad" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

void NPUSHMEMSymmetricMemory::barrier(int channel, size_t timeout_ms)
{
    // to be done
}

void NPUSHMEMSymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms)
{
    // to be done
}

void NPUSHMEMSymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms)
{
    // to be done
}

int NPUSHMEMSymmetricMemory::get_rank()
{
    return rank_;
}

int NPUSHMEMSymmetricMemory::get_world_size()
{
    return world_size_;
}

const std::vector<int>& NPUSHMEMSymmetricMemory::get_rank_to_global_rank()
{
    return rank_to_global_rank_;
}

int* NPUSHMEMSymmetricMemory::get_rank_to_global_rank_dev()
{
    return rank_to_global_rank_dev_;
}

void* NPUSHMEMSymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name)
{
    TORCH_CHECK(
        group_name == std::nullopt,
        "NPUSHMEMSymmetricMemoryAllocator::alloc "
        "must not be called with a group_name", DIST_ERROR(ErrCode::PARAM));
    logger->debug("NPUSHMEMSymmetricMemoryAllocator alloc start, size is %d, device is %d, group_name is %s",
        size, device_idx, group_name == std::nullopt ? "" : (*group_name).c_str());

    c10_npu::LazySetDevice(device_idx);
    auto group_info = get_group_info("0");
    auto store = group_info.store;
    int rank = group_info.rank;
    int world_size = group_info.world_size;
    npushmem_extension::initialize_npushmem_with_store(store, rank, world_size);

    auto ptr = Shmem_malloc(size);
    TORCH_CHECK(ptr != nullptr, "shmem_malloc return nullptr with size ", size, DIST_ERROR(ErrCode::MEMORY));
    auto allocation =
        std::make_shared<NPUSHMEMAllocation>(ptr, size, device_idx);
    // to be done: thread safety
    allocations_.try_emplace(ptr, std::move(allocation));
    logger->debug("NPUSHMEMSymmetricMemoryAllocator alloc end, size is %d, device is %d, group_name is %s, ptr is %p",
        size, device_idx, group_name == std::nullopt ? "" : (*group_name).c_str(), ptr);
    return ptr;
}

void NPUSHMEMSymmetricMemoryAllocator::free(void* ptr)
{
    logger->debug("NPUSHMEMSymmetricMemoryAllocator free start, ptr is %p", ptr);
    allocations_.erase(ptr);
    logger->debug("NPUSHMEMSymmetricMemoryAllocator free end, ptr is %p", ptr);
}

size_t NPUSHMEMSymmetricMemoryAllocator::get_alloc_size(void* ptr)
{
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        TORCH_CHECK(false, ptr, " is not allocated with NPUSHMEMSymmetricMemoryAllocator", DIST_ERROR(ErrCode::PARAM));
    }
    return it->second->buffer_size;
}

c10::intrusive_ptr<SymmetricMemory> NPUSHMEMSymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name)
{
    logger->debug("NPUSHMEMSymmetricMemoryAllocator rendezvous start, ptr is %p, group_name is %s", ptr, (*group_name).c_str());
    TORCH_CHECK(group_name.has_value(), "rendezvous, group_name is invalid.", DIST_ERROR(ErrCode::PARAM));
    {
        auto it = symm_mems_.find(std::make_tuple(ptr, *group_name));
        if (it != symm_mems_.end()) {
            return it->second;
        }
    }
    auto it = allocations_.find(ptr);
    TORCH_CHECK(it != allocations_.end(), "rendezvous, ptr is invalid.", DIST_ERROR(ErrCode::PARAM));
    auto symm_mem =
        c10::make_intrusive<NPUSHMEMSymmetricMemory>(it->second, *group_name);

    symm_mems_[std::make_tuple(ptr, *group_name)] = symm_mem;
    logger->debug("NPUSHMEMSymmetricMemoryAllocator rendezvous end, ptr is %p, group_name is %s", ptr, (*group_name).c_str());
    return symm_mem;
}

bool NPUSHMEMSymmetricMemoryAllocator::has_multicast_support(int device_idx)
{
    // to be done
    throw std::runtime_error("NPUSHMEMSymmetricMemoryAllocator does not support has_multicast_support" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

struct RegisterNPUSHMEMSymmetricMemoryAllocator {
    RegisterNPUSHMEMSymmetricMemoryAllocator()
    {
        register_allocator(
            c10::DeviceType::PrivateUse1,
            c10::make_intrusive<NPUSHMEMSymmetricMemoryAllocator>());
    }
};

static RegisterNPUSHMEMSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
