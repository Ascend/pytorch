#include <cstdint>
#include <memory>

#include <ATen/ceil_div.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSymmetricMemoryUtils.hpp"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMSymmetricMemory.hpp"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMSymmetricMemoryKernels.hpp"

namespace c10d {
namespace symmetric_memory {

#ifndef USE_NPU_SYMM_MEM
namespace {
[[noreturn]] void signal_kernels_unavailable() {
  TORCH_CHECK(
      false,
      "torch-npu was built without symmetric memory kernels. "
      "Rebuild with an available BiSheng compiler to enable barrier(), put_signal() and wait_signal().");
}
} // namespace

int32_t launch_npushmem_put_signal(void*, uint64_t, uint64_t, void*) {
  signal_kernels_unavailable();
}

int32_t launch_npushmem_wait_signal(void*, uint64_t, uint64_t, void*) {
  signal_kernels_unavailable();
}

size_t get_npushmem_barrier_workspace_size(int) {
  signal_kernels_unavailable();
}

int32_t launch_npushmem_barrier(void**, void*, int, int, int, uint64_t, void*) {
  signal_kernels_unavailable();
}
#endif

// Keep the data buffer aligned independently of the configured signal pad size.
constexpr size_t kSignalPadAlignment = 128;

/* Start of NPUSHMEMSymmetricMemory implementation */

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

  // Distributed initialization already registers the ProcessGroup; no separate
  // enable_symm_mem_for_group() call is needed to resolve its Store or ranks.
  auto group = resolve_process_group(group_name_);
  auto store = group->getStore();
  rank_ = group->getRank();
  world_size_ = group->getSize();
  auto global_rank = resolve_process_group("0")->getRank();
  // Exchange rank to global rank mapping for this group.
  // If it is already available, skip the exchange.
  std::lock_guard<std::mutex> rank_map_lock(rank_map_mutex);
  auto it = rank_to_global_rank_map.find(group_name_);
  if (it == rank_to_global_rank_map.end()) {
    // Each group's Store has its own namespace. Start this one-time exchange
    // locally so ranks that skipped another subgroup still use the same key.
    NPUStoreExchange exchange("NPUSHMEMSymmetricMemory");
    auto rank_to_global_rank = exchange.all_gather(store, rank_, world_size_, global_rank);
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

  // DataPtr members release the tables even if construction throws.
  const size_t arr_size = sizeof(void*) * world_size_;
  buffers_dev_owner_ = c10_npu::NPUCachingAllocator::get()->allocate(arr_size);
  TORCH_CHECK(
      buffers_dev_owner_, "NPUSHMEMSymmetricMemory failed to allocate buffers_dev_", DIST_ERROR(ErrCode::MEMORY));
  signal_pads_dev_owner_ = c10_npu::NPUCachingAllocator::get()->allocate(arr_size);
  TORCH_CHECK(
      signal_pads_dev_owner_,
      "NPUSHMEMSymmetricMemory failed to allocate signal_pads_dev_",
      DIST_ERROR(ErrCode::MEMORY));

  // The device pointer arrays must not alias the symmetric allocation.
  TORCH_CHECK(
      buffers_dev_owner_.get() != buffers_[rank_] && signal_pads_dev_owner_.get() != signal_pads_[rank_],
      "symm_mem device pointer array aliases the SHMEM allocation: buffers_dev_=",
      buffers_dev_owner_.get(),
      " signal_pads_dev_=",
      signal_pads_dev_owner_.get(),
      " buffer_ptr=",
      buffers_[rank_],
      DIST_ERROR(ErrCode::MEMORY));

  // Upload on the current stream, then sync: consumers may read these arrays
  // from another stream (pytorch/pytorch#192308).
  // stream() drains the host task queue so pending writes to reused allocator
  // blocks are enqueued before these copies. Reuse the ACL handle throughout.
  auto table_stream = c10_npu::getCurrentNPUStream(device_idx_).stream();
  NPU_CHECK_ERROR(aclrtMemcpyAsync(
      buffers_dev_owner_.get(), arr_size, buffers_.data(), arr_size, ACL_MEMCPY_HOST_TO_DEVICE, table_stream));
  NPU_CHECK_ERROR(aclrtMemcpyAsync(
      signal_pads_dev_owner_.get(), arr_size, signal_pads_.data(), arr_size, ACL_MEMCPY_HOST_TO_DEVICE, table_stream));
  NPU_CHECK_ERROR(aclrtSynchronizeStream(table_stream));

  buffers_dev_ = static_cast<void**>(buffers_dev_owner_.get());
  signal_pads_dev_ = static_cast<void**>(signal_pads_dev_owner_.get());

  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemory created, buffer_size is %d, device_idx is %d, group_name is %s.",
      allocation->buffer_size,
      allocation->device_idx,
      group_name_.c_str());
}

NPUSHMEMSymmetricMemory::~NPUSHMEMSymmetricMemory() {
  if (is_finalizing()) {
    // Returning alone still invokes the DataPtr deleters. Release their contexts
    // to avoid allocator calls after the driver has shut down.
    (void)buffers_dev_owner_.release_context();
    (void)signal_pads_dev_owner_.release_context();
    return;
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
  const auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx_);
  at::DeviceGuard device_guard(device);
  // Call on each consuming stream. This records only the table, not its SHMEM entries.
  c10_npu::NPUCachingAllocator::recordStream(buffers_dev_owner_, c10_npu::getCurrentNPUStream(device_idx_));
  return buffers_dev_;
}

void** NPUSHMEMSymmetricMemory::get_signal_pad_ptrs_dev() {
  const auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx_);
  at::DeviceGuard device_guard(device);
  // Register every stream that consumes this table, as for the buffer table.
  c10_npu::NPUCachingAllocator::recordStream(signal_pads_dev_owner_, c10_npu::getCurrentNPUStream(device_idx_));
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

static void check_signal_channel(int channel, int world_size) {
  TORCH_CHECK(
      channel >= 0,
      "channel for barrier(), put_signal() and wait_signal() must be non-negative (got ",
      channel,
      ")",
      DIST_ERROR(ErrCode::PARAM));
  const size_t num_channels = get_signal_pad_size() / (sizeof(uint32_t) * world_size);
  TORCH_CHECK(
      static_cast<size_t>(channel) < num_channels,
      "The maximum supported channel for barrier(), put_signal() and wait_signal() is ",
      static_cast<int64_t>(num_channels) - 1,
      " (got ",
      channel,
      ")",
      DIST_ERROR(ErrCode::PARAM));
}

void NPUSHMEMSymmetricMemory::barrier(int channel, size_t timeout_ms) {
  check_signal_channel(channel, world_size_);
  const auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx_);
  at::DeviceGuard device_guard(device);
  const auto npu_stream = c10_npu::getCurrentNPUStream(device_idx_);
  auto stream = npu_stream.stream();
  TORCH_CHECK(stream != nullptr, "barrier: failed to obtain the current NPU stream", DIST_ERROR(ErrCode::INTERNAL));

  // Each invocation owns its progress state; the kernel initializes it before
  // use. Separate scratch does not isolate overlapping signal-pad slots.
  auto peer_states = c10_npu::NPUCachingAllocator::get()->allocate(get_npushmem_barrier_workspace_size(world_size_));
  // Keep the scratch and pointer table alive until the enqueued kernel is done.
  c10_npu::NPUCachingAllocator::recordStream(peer_states, npu_stream);
  c10_npu::NPUCachingAllocator::recordStream(signal_pads_dev_owner_, npu_stream);
  NPU_CHECK_ERROR(
      launch_npushmem_barrier(signal_pads_dev_, peer_states.get(), channel, rank_, world_size_, timeout_ms, stream));
}

void NPUSHMEMSymmetricMemory::put_signal(int dst_rank, int channel, size_t timeout_ms) {
  TORCH_CHECK(
      dst_rank >= 0 && dst_rank < world_size_,
      "put_signal: destination rank must be in [0, ",
      world_size_,
      "), got ",
      dst_rank,
      DIST_ERROR(ErrCode::PARAM));
  check_signal_channel(channel, world_size_);

  // The mapped peer pad already accounts for the SHMEM WORLD PE; slots use
  // group-local ranks, independently of that global mapping.
  const uint64_t signal_index = static_cast<uint64_t>(channel) * world_size_ + rank_;
  const auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx_);
  at::DeviceGuard device_guard(device);
  auto stream = c10_npu::getCurrentNPUStream(device_idx_).stream();
  TORCH_CHECK(stream != nullptr, "put_signal: failed to obtain the current NPU stream", DIST_ERROR(ErrCode::INTERNAL));

  // Only enqueue: CAS backpressure and timeout handling run on the device.
  NPU_CHECK_ERROR(launch_npushmem_put_signal(signal_pads_[dst_rank], signal_index, timeout_ms, stream));
}

void NPUSHMEMSymmetricMemory::wait_signal(int src_rank, int channel, size_t timeout_ms) {
  TORCH_CHECK(
      src_rank >= 0 && src_rank < world_size_,
      "wait_signal: source rank must be in [0, ",
      world_size_,
      "), got ",
      src_rank,
      DIST_ERROR(ErrCode::PARAM));
  check_signal_channel(channel, world_size_);

  // Consume the source's slot in our local pad, without a separate clear task.
  const uint64_t signal_index = static_cast<uint64_t>(channel) * world_size_ + src_rank;
  const auto device = c10::Device(at::DeviceType::PrivateUse1, device_idx_);
  at::DeviceGuard device_guard(device);
  auto stream = c10_npu::getCurrentNPUStream(device_idx_).stream();
  TORCH_CHECK(stream != nullptr, "wait_signal: failed to obtain the current NPU stream", DIST_ERROR(ErrCode::INTERNAL));

  NPU_CHECK_ERROR(launch_npushmem_wait_signal(signal_pads_[rank_], signal_index, timeout_ms, stream));
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

  // Resolve an unindexed "npu" device before selecting it or storing its index.
  if (device_idx == -1) {
    device_idx = c10_npu::current_device();
  }
  c10_npu::LazySetDevice(device_idx);
  // SHMEM bootstraps over WORLD even when a later rendezvous uses a subgroup.
  auto group = resolve_process_group("0");
  npushmem_extension::initialize_npushmem_with_store(group->getStore(), group->getRank(), group->getSize());

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
  TORCH_CHECK(group_name.has_value(), "rendezvous, group_name is invalid.", DIST_ERROR(ErrCode::PARAM));
  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemoryAllocator rendezvous start, ptr is %p, group_name is %s", ptr, (*group_name).c_str());
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
