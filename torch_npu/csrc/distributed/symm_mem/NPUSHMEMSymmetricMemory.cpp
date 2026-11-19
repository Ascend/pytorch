#include <memory>

#include <ATen/ceil_div.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSymmetricMemoryUtils.hpp"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMSymmetricMemory.hpp"

namespace c10d {
namespace symmetric_memory {

// Keep the data buffer aligned independently of the configured signal pad size.
constexpr size_t kSignalPadAlignment = 128;

// RAII guard for a caching-allocator device pointer. The constructor allocates
// buffers_dev_/signal_pads_dev_ before the object is fully constructed; if a
// later step (raw_alloc/aclrtMemcpy for the second array) throws via
// TORCH_CHECK, the destructor will NOT run, so the first array would leak.
// Hold each array under this guard and release() to the member only after both
// succeed, so any exception on the way frees whatever was already allocated.
struct DevPtrDeleter {
  void operator()(void* p) const noexcept {
    if (p != nullptr) {
      c10_npu::NPUCachingAllocator::raw_delete(p);
    }
  }
};
using DevPtrGuard = std::unique_ptr<void, DevPtrDeleter>;

/* Start of NPUSHMEMSymmetricMemory implementation */

static NPUStoreExchange storeExchange = NPUStoreExchange("NPUSHMEMSymmetricMemory");

static std::mutex rank_map_mutex;
static std::unordered_map<std::string, std::vector<int>> rank_to_global_rank_map{};

NPUSHMEMAllocation::~NPUSHMEMAllocation() {
  // Avoid calling NPU functions after driver shutting down
  if (is_finalizing()) {
    return;
  }
  auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx);
  at::DeviceGuard device_guard(device);
  TORCH_NPU_SYMMEM_LOGD("~NPUSHMEMAllocation, start Shmem_free, ptr is %p.", alloc_base);
  Aclshmem_free(alloc_base); // shmem_free has no return value
  TORCH_NPU_SYMMEM_LOGD("~NPUSHMEMAllocation, end Shmem_free, ptr is %p.", alloc_base);
}

NPUSHMEMSymmetricMemory::NPUSHMEMSymmetricMemory(
    std::shared_ptr<NPUSHMEMAllocation> allocation,
    const std::string& group_name)
    : allocation_(allocation),
      buffer_size_(allocation->buffer_size),
      device_idx_(allocation->device_idx),
      group_name_(group_name) {
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
  std::lock_guard<std::mutex> rank_map_lock(rank_map_mutex);
  auto it = rank_to_global_rank_map.find(group_name_);
  if (it == rank_to_global_rank_map.end()) {
    auto rank_to_global_rank = storeExchange.all_gather(store, rank_, world_size_, global_rank);
    exchanged_n_times++;
    if (rank_ == 0) {
      std::stringstream ss;
      for (size_t i = 0; i < rank_to_global_rank.size(); ++i) {
        ss << rank_to_global_rank[i];
        if (i != rank_to_global_rank.size() - 1) {
          ss << ", ";
        }
      }
      TORCH_NPU_SYMMEM_LOGD(
          "[rank %d] rank_to_global_rank: %s, group_name: %s, exchanged_n_times: %d.",
          rank_,
          (ss.str()).c_str(),
          group_name_.c_str(),
          exchanged_n_times);
    }
    it = rank_to_global_rank_map.emplace_hint(it, group_name_, rank_to_global_rank);
  }
  auto& rank_to_global_rank = it->second;
  for (int r = 0; r < world_size_; ++r) {
    auto peer_base = Aclshmem_ptr(allocation->alloc_base, rank_to_global_rank[r]);
    TORCH_CHECK(
        peer_base != nullptr,
        "shmem_ptr return nullptr with ptr ",
        allocation->alloc_base,
        DIST_ERROR(ErrCode::MEMORY));
    signal_pads_.push_back(peer_base);
    buffers_.push_back(static_cast<char*>(peer_base) + allocation->buffer_offset);
    TORCH_NPU_SYMMEM_LOGD(
        "[rank %d] NPUSHMEMSymmetricMemory shmem_ptr, r is %d, rank_to_global_rank is %d, ptr is %p, shmem_ptr is %p.",
        rank_,
        r,
        rank_to_global_rank[r],
        allocation->alloc_base,
        peer_base);
  }

  // Copy the per-rank buffer pointers to device memory so device-side kernels
  // can access them via get_buffer_ptrs_dev().
  const size_t arr_size = sizeof(void*) * world_size_;
  DevPtrGuard buffers_dev_guard(c10_npu::NPUCachingAllocator::raw_alloc(arr_size));
  TORCH_CHECK(
      buffers_dev_guard, "NPUSHMEMSymmetricMemory failed to allocate buffers_dev_", DIST_ERROR(ErrCode::MEMORY));
  DevPtrGuard signal_pads_dev_guard(c10_npu::NPUCachingAllocator::raw_alloc(arr_size));
  TORCH_CHECK(
      signal_pads_dev_guard,
      "NPUSHMEMSymmetricMemory failed to allocate signal_pads_dev_",
      DIST_ERROR(ErrCode::MEMORY));

  // The device pointer arrays must not alias the symmetric allocation.
  TORCH_CHECK(
      buffers_dev_guard.get() != buffers_[rank_] && signal_pads_dev_guard.get() != signal_pads_[rank_],
      "symm_mem device pointer array aliases the SHMEM allocation: buffers_dev_=",
      buffers_dev_guard.get(),
      " signal_pads_dev_=",
      signal_pads_dev_guard.get(),
      " buffer_ptr=",
      buffers_[rank_],
      DIST_ERROR(ErrCode::MEMORY));

  // Upload on the current stream, then sync: consumers may read these arrays
  // from another stream (pytorch/pytorch#192308).
  auto stream = c10_npu::getCurrentNPUStream(static_cast<c10::DeviceIndex>(device_idx_));
  NPU_CHECK_ERROR(aclrtMemcpyAsync(
      buffers_dev_guard.get(), arr_size, buffers_.data(), arr_size, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  NPU_CHECK_ERROR(aclrtMemcpyAsync(
      signal_pads_dev_guard.get(), arr_size, signal_pads_.data(), arr_size, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  NPU_CHECK_ERROR(aclrtSynchronizeStream(stream));

  // Both succeeded: transfer ownership to members (destructor will raw_delete).
  buffers_dev_ = reinterpret_cast<void**>(buffers_dev_guard.release());
  signal_pads_dev_ = reinterpret_cast<void**>(signal_pads_dev_guard.release());

  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemory created, buffer_size is %d, device_idx is %d, group_name is %s.",
      allocation->buffer_size,
      allocation->device_idx,
      group_name_.c_str());
}

NPUSHMEMSymmetricMemory::~NPUSHMEMSymmetricMemory() {
  if (buffers_dev_ != nullptr) {
    c10_npu::NPUCachingAllocator::raw_delete(buffers_dev_);
    buffers_dev_ = nullptr;
  }
  if (signal_pads_dev_ != nullptr) {
    c10_npu::NPUCachingAllocator::raw_delete(signal_pads_dev_);
    signal_pads_dev_ = nullptr;
  }
  TORCH_NPU_SYMMEM_LOGD("NPUSHMEMSymmetricMemory destroy, group_name is %s", group_name_.c_str());
}

std::vector<void*> NPUSHMEMSymmetricMemory::get_buffer_ptrs() {
  return buffers_;
}

std::vector<void*> NPUSHMEMSymmetricMemory::get_signal_pad_ptrs() {
  return signal_pads_;
}

void** NPUSHMEMSymmetricMemory::get_buffer_ptrs_dev() {
  return buffers_dev_;
}

void** NPUSHMEMSymmetricMemory::get_signal_pad_ptrs_dev() {
  return signal_pads_dev_;
}

size_t NPUSHMEMSymmetricMemory::get_buffer_size() {
  return buffer_size_;
}

bool NPUSHMEMSymmetricMemory::has_multicast_support() {
  return false;
}

void* NPUSHMEMSymmetricMemory::get_multicast_ptr() {
  return nullptr;
}

void NPUSHMEMSymmetricMemory::barrier(int channel, size_t timeout_ms) {
  // to be done
  throw std::runtime_error("NPUSHMEMSymmetricMemory does not support barrier" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

void NPUSHMEMSymmetricMemory::put_signal(int dst_rank, int channel, size_t timeout_ms) {
  // to be done
  throw std::runtime_error("NPUSHMEMSymmetricMemory does not support put_signal" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

void NPUSHMEMSymmetricMemory::wait_signal(int src_rank, int channel, size_t timeout_ms) {
  // to be done
  throw std::runtime_error("NPUSHMEMSymmetricMemory does not support wait_signal" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

int NPUSHMEMSymmetricMemory::get_rank() {
  return rank_;
}

int NPUSHMEMSymmetricMemory::get_world_size() {
  return world_size_;
}

const std::vector<int>& NPUSHMEMSymmetricMemory::get_rank_to_global_rank() {
  std::lock_guard<std::mutex> lock(rank_map_mutex);
  auto it = rank_to_global_rank_map.find(group_name_);
  TORCH_CHECK(
      it != rank_to_global_rank_map.end(),
      "Group name not found in rank_to_global_rank_map",
      DIST_ERROR(ErrCode::PARAM));
  return it->second;
}

c10::Device NPUSHMEMSymmetricMemory::get_device() {
  return c10::Device(at::DeviceType::PrivateUse1, device_idx_);
}

void* NPUSHMEMSymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name) {
  TORCH_CHECK(
      group_name == std::nullopt,
      "NPUSHMEMSymmetricMemoryAllocator::alloc "
      "must not be called with a group_name",
      DIST_ERROR(ErrCode::PARAM));
  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemoryAllocator alloc start, size is %d, device is %d, group_name is %s",
      size,
      device_idx,
      group_name == std::nullopt ? "" : (*group_name).c_str());

  c10_npu::LazySetDevice(device_idx);
  auto group_info = get_group_info("0");
  auto store = group_info.store;
  int rank = group_info.rank;
  int world_size = group_info.world_size;
  npushmem_extension::initialize_npushmem_with_store(store, rank, world_size);

  // Layout: [signal pad | alignment padding | data buffer]. The data offset
  // depends only on the configured pad size, never on the requested data size.
  const size_t buffer_offset = at::round_up(get_signal_pad_size(), kSignalPadAlignment);
  const size_t alloc_size = buffer_offset + at::round_up(size, size_t{16});
  auto alloc_base = Aclshmem_malloc(alloc_size);
  TORCH_CHECK(alloc_base != nullptr, "shmem_malloc return nullptr with size ", alloc_size, DIST_ERROR(ErrCode::MEMORY));
  // Initialize the entire prefix once, then wait until every rank has finished
  // zeroing its pad before any peer can signal into this allocation.
  auto stream = c10_npu::getCurrentNPUStream(static_cast<c10::DeviceIndex>(device_idx));
  auto ret = c10_npu::acl::AclrtMemSetAsync(alloc_base, buffer_offset, 0, buffer_offset, stream);
  TORCH_CHECK(
      ret == ACL_ERROR_NONE, "failed to initialize NPUSHMEM signal pad, ret is ", ret, DIST_ERROR(ErrCode::INTERNAL));
  NPU_CHECK_ERROR(aclrtSynchronizeStream(stream));
  Aclshmem_barrier_all();
  void* buffer_ptr = static_cast<char*>(alloc_base) + buffer_offset;
  auto allocation = std::make_shared<NPUSHMEMAllocation>(alloc_base, size, buffer_offset, device_idx);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // free(), get_alloc_size(), and rendezvous() receive the returned data ptr.
    allocations_.try_emplace(buffer_ptr, std::move(allocation));
  }
  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemoryAllocator alloc end, size is %d, device is %d, group_name is %s, ptr is %p",
      size,
      device_idx,
      group_name == std::nullopt ? "" : (*group_name).c_str(),
      buffer_ptr);
  return buffer_ptr;
}

void NPUSHMEMSymmetricMemoryAllocator::free(void* ptr) {
  TORCH_NPU_SYMMEM_LOGD("NPUSHMEMSymmetricMemoryAllocator free start, ptr is %p", ptr);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto alloc_it = allocations_.find(ptr);
    if (alloc_it == allocations_.end()) {
      return;
    }
    for (auto it = symm_mems_.begin(); it != symm_mems_.end();) {
      if (std::get<0>(it->first) == ptr) {
        it = symm_mems_.erase(it);
      } else {
        ++it;
      }
    }
    allocations_.erase(alloc_it);
  }
  TORCH_NPU_SYMMEM_LOGD("NPUSHMEMSymmetricMemoryAllocator free end, ptr is %p", ptr);
}

size_t NPUSHMEMSymmetricMemoryAllocator::get_alloc_size(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    TORCH_CHECK(false, ptr, " is not allocated with NPUSHMEMSymmetricMemoryAllocator", DIST_ERROR(ErrCode::PARAM));
  }
  return it->second->buffer_size;
}

c10::intrusive_ptr<SymmetricMemory> NPUSHMEMSymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name) {
  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemoryAllocator rendezvous start, ptr is %p, group_name is %s", ptr, (*group_name).c_str());
  TORCH_CHECK(group_name.has_value(), "rendezvous, group_name is invalid.", DIST_ERROR(ErrCode::PARAM));
  std::lock_guard<std::mutex> lock(mutex_);
  {
    auto it = symm_mems_.find(std::make_tuple(ptr, *group_name));
    if (it != symm_mems_.end()) {
      return it->second;
    }
  }
  auto it = allocations_.find(ptr);
  TORCH_CHECK(it != allocations_.end(), "rendezvous, ptr is invalid.", DIST_ERROR(ErrCode::PARAM));
  auto symm_mem = c10::make_intrusive<NPUSHMEMSymmetricMemory>(it->second, *group_name);

  symm_mems_[std::make_tuple(ptr, *group_name)] = symm_mem;
  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemoryAllocator rendezvous end, ptr is %p, group_name is %s", ptr, (*group_name).c_str());
  return symm_mem;
}

bool NPUSHMEMSymmetricMemoryAllocator::has_multicast_support(int device_idx) {
  // to be done
  throw std::runtime_error(
      "NPUSHMEMSymmetricMemoryAllocator does not support has_multicast_support" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

c10::DeviceType NPUSHMEMSymmetricMemoryAllocator::supported_device_type() {
  return c10::DeviceType::PrivateUse1;
}

std::string NPUSHMEMSymmetricMemoryAllocator::name() {
  return "NPUSHMEM";
}

size_t NPUSHMEMSymmetricMemory::get_offset() {
  return 0;
}

struct RegisterNPUSHMEMSymmetricMemoryAllocator {
  RegisterNPUSHMEMSymmetricMemoryAllocator() {
    register_allocator(c10::DeviceType::PrivateUse1, c10::make_intrusive<NPUSHMEMSymmetricMemoryAllocator>());
  }
};

static RegisterNPUSHMEMSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
