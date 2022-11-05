// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <algorithm>
#include <bitset>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <c10/util/UniqueVoidPtr.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace c10_npu {
namespace NPUCachingAllocator {

C10_DEFINE_REGISTRY(FreeNPUMemoryCallbacksRegistry, FreeMemoryCallback);

//
// Yet another caching allocator for NPU device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to npuMalloc.
// - If the npuMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using npuMalloc.
//   To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//
namespace {
using stream_set = std::unordered_set<c10_npu::NPUStream>;

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocs to 2 MiB

using StatTypes = std::bitset<static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;
  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

void update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for (size_t stat_type = 0; stat_type < stat_types.size(); ++stat_type) {
    if (stat_types[stat_type]) {
      update_stat(stat_array[stat_type], amount);
    }
  }
}

struct DeviceStats {
  uint64_t amount_allocated; // total amount allocated in bytes
  uint64_t max_amount_allocated; // max total amount allocated in bytes
  uint64_t amount_cached; // total amount in cache in bytes
  uint64_t max_amount_cached; // max total amount in cache in bytes

  DeviceStats()
      : amount_allocated(0),
        max_amount_allocated(0),
        amount_cached(0),
        max_amount_cached(0) {}

  void increaseAllocated(size_t delta) {
    amount_allocated += delta;
    max_amount_allocated = std::max(max_amount_allocated, amount_allocated);
  }

  void decreaseAllocated(size_t delta) {
    amount_allocated -= delta;
  }

  void increaseCached(size_t delta) {
    amount_cached += delta;
    max_amount_cached = std::max(max_amount_cached, amount_cached);
  }

  void decreaseCached(size_t delta) {
    amount_cached -= delta;
  }
};

struct Block;
using Comparison = bool (*)(const Block*, const Block*);
using BlockPool = std::set<Block*, Comparison>;

struct Block {
  int device; // npu
  aclrtStream stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  BlockPool* pool; // owning memory pool
  void* ptr; // memory address
  bool allocated; // in-use flag
  Block* prev; // prev block if split from a larger allocation
  Block* next; // next block if split from a larger allocation
  int event_count; // number of outstanding NPU events

  Block(int device, aclrtStream stream, size_t size, BlockPool* pool, void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        pool(pool),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        event_count(0) {}

  // constructor for search key
  Block(int device, aclrtStream stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        pool(nullptr),
        ptr(nullptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        event_count(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->device != b->device) {
    return a->device < b->device;
  }
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
        reinterpret_cast<uintptr_t>(b->stream);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}

static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << (size / 1048576.0);
    os << " MiB";
  } else {
    os << (size / 1073741824.0);
    os << " GiB";
  }
  return os.str();
}
} // namespace

struct THNCachingAllocator {
  // device statistics
  std::vector<DeviceStats> device_stats;
  std::vector<DeviceStats_> device_stats_;

  // lock around all operations
  mutable std::recursive_mutex mutex;

  // lock around calls to aclFree (to prevent deadlocks with NCCL)
  mutable std::mutex npu_free_mutex;

  mutable std::mutex recorded_event_mutex;

  // cached blocks larger than 1 MB
  BlockPool large_blocks;

  // cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // outstanding acl events
  std::deque<std::pair<aclrtEvent, Block*>> npu_events;

  std::set<aclrtEvent> recorded_events;

  THNCachingAllocator()
      : large_blocks(BlockComparator), small_blocks(BlockComparator) {}

  DeviceStats& get_stats_for_device(int device) {
    AT_ASSERT(device >= 0);
    if ((size_t)device >= device_stats.size()) {
      device_stats.resize(device + 1);
    }
    return device_stats.at(device);
  }

  DeviceStats_& get_stats_for_device_(int device) {
    AT_ASSERT(device >= 0);
    if ((size_t)device >= device_stats_.size()) {
      device_stats_.resize(device + 1);
    }
    return device_stats_.at(device);
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, size_t size, aclrtStream stream, int device = -1);

  void free(void* ptr) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!ptr) {
      return;
    }

    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      AT_ERROR("invalid device pointer: ", ptr);
    }

    Block* block = it->second;
    allocated_blocks.erase(it);
    block->allocated = false;

    c10::reportMemoryUsageToProfiler(
        block, -block->size, c10::Device(at_npu::key::NativeDeviceType, block->device));

    DeviceStats_& stats_ = get_stats_for_device_(block->device);
    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] =
        true;
    update_stat_array(stats_.allocation, -1, {stat_types});
    update_stat_array(stats_.allocated_bytes, -block->size, {stat_types});
    get_stats_for_device(block->device).decreaseAllocated(block->size);

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }
  }

  /** returns cached blocks to the system allocator */
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    synchronize_and_free_events(c10::nullopt);
    c10_npu::npuSynchronizeDevice();
    free_blocks(large_blocks, large_blocks.begin(), large_blocks.end());
    free_blocks(small_blocks, small_blocks.begin(), small_blocks.end());
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    Block* block = find_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cacheInfoAux(
      const BlockPool& blocks,
      int dev_id,
      size_t* total,
      size_t* largest) {
    Block search_key(dev_id, 0, 0);
    auto it = blocks.lower_bound(&search_key);
    for (; it != blocks.end() && *it && (*it)->device == dev_id; ++it) {
      size_t blocksize = (*it)->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void cacheInfo(int dev_id, size_t* total, size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cacheInfoAux(large_blocks, dev_id, total, largest);
    cacheInfoAux(small_blocks, dev_id, total, largest);
  }

  /** Returns a copy of the memory allocator stats for the device **/
  DeviceStats_ getStatsForDevice(int dev_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return get_stats_for_device_(dev_id);
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats(int dev_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    DeviceStats_& stats = get_stats_for_device_(dev_id);

    for (size_t statType = 0;
         statType < static_cast<size_t>(StatType::NUM_TYPES);
         ++statType) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats(int dev_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    DeviceStats_& stats = get_stats_for_device_(dev_id);

    for (size_t statType = 0;
         statType < static_cast<size_t>(StatType::NUM_TYPES);
         ++statType) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
    }
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
      if (head_block->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<uintptr_t>(head_block->ptr);
      segment_info.is_large = (head_block->pool == &large_blocks);

      const Block* block = head_block;
      while (block != nullptr) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0);

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
        }

        block = block->next;
      }
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          if (a.device != b.device) {
            return a.device < b.device;
          }
          return a.address < b.address;
        });

    return result;
  }

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(blocks.end(), small_blocks.begin(), small_blocks.end());
    blocks.insert(blocks.end(), large_blocks.begin(), large_blocks.end());
    for (const auto& item : allocated_blocks) {
      blocks.push_back(item.second);
    }
    return blocks;
  }

  void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }
    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when NPU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &raw_delete) {
      return;
    }
    std::lock_guard<std::recursive_mutex> lock(mutex);
    Block* block = find_allocated_block(ptr.get());
    // block could be nullptr in some cases, e.g., tensor loaded from blob, or
    // shared from another process, or not pointing to a NPU tensor.
    if (block) {
      if (stream.stream() == block->stream) {
        // ignore uses on the allocation stream, since those don't require any
        // special synchronization
        return;
      }
      block->stream_uses.insert(stream);
    }
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block) {
    AT_ASSERT(!block->allocated && block->event_count == 0);
    size_t original_block_size = block->size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const int64_t subsumed_size_prev =
        try_merge_blocks(block, block->prev, pool);
    if (subsumed_size_prev > 0) {
      net_change_inactive_split_blocks -= 1;
      net_change_inactive_split_size -= subsumed_size_prev;
    }
    const int64_t subsumed_size_next =
        try_merge_blocks(block, block->next, pool);
    if (subsumed_size_next > 0) {
      net_change_inactive_split_blocks -= 1;
      net_change_inactive_split_size -= subsumed_size_next;
    }
    pool.insert(block);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    DeviceStats_& stats_ = get_stats_for_device_(block->device);
    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] =
        true;

    update_stat_array(
        stats_.inactive_split, net_change_inactive_split_blocks, stat_types);
    update_stat_array(
        stats_.inactive_split_bytes,
        net_change_inactive_split_size,
        stat_types);
    update_stat_array(stats_.active, -1, stat_types);
    update_stat_array(stats_.active_bytes, -original_block_size, stat_types);
  }

  /** combine previously split blocks */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0) {
      return 0;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += src->size;
    pool.erase(src);
    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatType get_stat_type_for_pool(const BlockPool& pool) {
    if (&pool == &small_blocks) {
      return StatType::SMALL_POOL;
    } else if (&pool == &large_blocks) {
      return StatType::LARGE_POOL;
    } else {
      AT_ERROR("get_stat_type_for_pool: invalid pool");
    }
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool == &small_blocks) {
      return remaining >= kMinBlockSize;
    } else if (block->pool == &large_blocks) {
      return remaining > kSmallSize;
    } else {
      AT_ERROR("should_split: invalid pool");
    }
  }

  size_t round_size(size_t size) {
    // be consistent with ACL memory alloc rules
    size = size + 32;
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  aclError npu_malloc_retry(int device, void** devPtr, size_t size) {
    // Try npuMalloc. If npuMalloc fails, frees all non-split cached blocks
    // and retries.
    aclError err = aclrtMalloc(
        devPtr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_ERROR_NONE) {
      DeviceStats_& stats_ = get_stats_for_device_(device);
      stats_.num_alloc_retries += 1;

      // npuGetLastError();  // reset the last NPU error
      free_cached_blocks(device);
      err = aclrtMalloc(
          devPtr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
      if (err != ACL_ERROR_NONE) {
        C10_NPU_SHOW_ERR_MSG();
        return err;
      }
    }
    return ACL_ERROR_NONE;
  }

  void free_cached_blocks(int device) {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(device);

    // Free all non-split cached blocks on device
    Block lower_bound(device, nullptr, 0);
    Block upper_bound(device + 1, nullptr, 0);

    c10_npu::npuSynchronizeDevice();
    free_blocks(
        large_blocks,
        large_blocks.lower_bound(&lower_bound),
        large_blocks.lower_bound(&upper_bound));
    free_blocks(
        small_blocks,
        small_blocks.lower_bound(&lower_bound),
        small_blocks.lower_bound(&upper_bound));
  }

  void free_blocks(
      BlockPool& blocks,
      BlockPool::iterator it,
      BlockPool::iterator end) {
    // Frees all non-split blocks between `it` and `end`
    std::lock_guard<std::mutex> lock(npu_free_mutex);
    while (it != end) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        aclrtFree((void*)block->ptr);

        get_stats_for_device(block->device).decreaseCached(block->size);
        DeviceStats_& stats_ = get_stats_for_device_(block->device);
        StatTypes stat_types;
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(
            get_stat_type_for_pool(*(block->pool)))] = true;

        update_stat_array(stats_.segment, -1, stat_types);
        update_stat_array(stats_.reserved_bytes, -block->size, stat_types);

        auto cur = it;
        ++it;
        blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
  }

  void synchronize_and_free_events(c10::optional<int> device) {
    // Synchronize on outstanding events and then free associated blocks.
    // Limited to blocks on the given device if specified.
    auto remaining_events = decltype(npu_events)();

    for (auto& e : npu_events) {
      aclrtEvent event = e.first;
      {
        std::lock_guard<std::mutex> lock(recorded_event_mutex);
        auto it = recorded_events.find(event);
        if (c10_npu::option::OptionsManager::CheckQueueEnable() &&
            it == recorded_events.end()) {
          break;
        }
      }
      Block* block = e.second;
      if (device.has_value() && block->device != *device) {
        remaining_events.push_back(e);
        continue;
      }

      C10_NPU_CHECK(aclrtSynchronizeEvent(event));
      {
        std::lock_guard<std::mutex> lock(recorded_event_mutex);
        auto it = recorded_events.find(event);
        if (it != recorded_events.end()) {
          recorded_events.erase(it);
        }
      }
      C10_NPU_CHECK(aclrtDestroyEvent(event));
      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    std::swap(npu_events, remaining_events);
  }

  Block* find_allocated_block(void* ptr) {
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    return it->second;
  }

  void insertRecordedEvent(aclrtEvent event) {
    std::lock_guard<std::mutex> lock(recorded_event_mutex);
    recorded_events.insert(event);
  }

  void insert_events(Block* block) {
    int prev_device = 0;
    C10_NPU_CHECK(aclrtGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      int pre_device = 0;
      aclError ret = aclrtGetDevice(&pre_device);
      if (ret != ACL_ERROR_NONE) {
        C10_NPU_CHECK(aclrtSetDevice(it->device_index()));
      } else if (pre_device != it->device_index()) {
        C10_NPU_CHECK(aclrtSetDevice(it->device_index()));
      }

      aclrtEvent event = nullptr;
      C10_NPU_CHECK(c10_npu::acl::AclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE));
      c10_npu::queue::NpuAllocatorLaunchRecordEventTask(event, *it);

      block->event_count++;
      npu_events.emplace_back(event, block);
    }

    int cur_device = 0;
    aclError ret = aclrtGetDevice(&cur_device);
    if (ret != ACL_ERROR_NONE) {
      C10_NPU_CHECK(aclrtSetDevice(prev_device));
    } else if (cur_device != prev_device) {
      C10_NPU_CHECK(aclrtSetDevice(prev_device));
    }
  }

  void process_events() {
    // Process outstanding npuEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!npu_events.empty()) {
      auto& e = npu_events.front();
      aclrtEvent event = e.first;
      Block* block = e.second;

      {
        std::lock_guard<std::mutex> lock(recorded_event_mutex);
        auto it = recorded_events.begin();
        it = recorded_events.find(event);
        if (c10_npu::option::OptionsManager::CheckQueueEnable() &&
            it == recorded_events.end()) {
          break;
        }
      }

      c10_npu::acl::aclrtEventRecordedStatus status =
          c10_npu::acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
      aclError err = c10_npu::acl::AclQueryEventRecordedStatus(event, &status);
      if (err != ACL_ERROR_NONE) {
           C10_NPU_CHECK(err);
      }
      if (status != c10_npu::acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
        break;
      }

      {
        std::lock_guard<std::mutex> lock(recorded_event_mutex);
        auto it = recorded_events.find(event);
        if (it != recorded_events.end()) {
          recorded_events.erase(it);
        }
      }
      C10_NPU_CHECK(aclrtDestroyEvent(event));

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      npu_events.pop_front();
    }
  }

  void allocate_adjacent_ptr(
      size_t size1,
      size_t size2,
      void** ptr_pre,
      void** ptr_next,
      aclrtStream stream) {
    size_t round_size_pre = (size1 + 32 + 511) / 512 * 512;
    size_t round_size = round_size_pre + size2;
    malloc(ptr_pre, round_size, stream);

    Block* temp_block = allocated_blocks.find(*ptr_pre)->second;
    DeviceStats_& stats_ = get_stats_for_device_(temp_block->device);
    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(temp_block->pool)))] = true;
    update_stat_array(stats_.allocation, -1, {stat_types});
    update_stat_array(stats_.allocated_bytes, -temp_block->size, {stat_types});
    update_stat_array(stats_.active, -1, {stat_types});
    update_stat_array(stats_.active_bytes, -temp_block->size, {stat_types});

    Block* next_block = nullptr;
    Block* pre_block = allocated_blocks.find(*ptr_pre)->second;
    next_block = pre_block;
    auto& pool = get_pool(round_size);
    pre_block = new Block(
        next_block->device,
        next_block->stream,
        round_size_pre,
        &pool,
        pre_block->ptr);

    pre_block->prev = next_block->prev;
    if (pre_block->prev) {
      pre_block->prev->next = pre_block;
    }
    pre_block->next = next_block;
    next_block->prev = pre_block;
    next_block->ptr = static_cast<char*>(next_block->ptr) + round_size_pre;
    pre_block->size = round_size_pre;
    next_block->size -= round_size_pre;

    pre_block->allocated = true;
    next_block->allocated = true;
    allocated_blocks[pre_block->ptr] = pre_block;
    allocated_blocks[next_block->ptr] = next_block;

    *ptr_next = next_block->ptr;

    DeviceStats_& stats_pre = get_stats_for_device_(pre_block->device);
    StatTypes stat_types_pre;
    stat_types_pre[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types_pre[static_cast<size_t>(
        get_stat_type_for_pool(*(pre_block->pool)))] = true;
    update_stat_array(stats_pre.allocation, 1, stat_types_pre);
    update_stat_array(
        stats_pre.allocated_bytes, pre_block->size, stat_types_pre);
    update_stat_array(stats_pre.active, 1, stat_types_pre);
    update_stat_array(stats_pre.active_bytes, pre_block->size, stat_types_pre);

    DeviceStats_& stats_next = get_stats_for_device_(next_block->device);
    StatTypes stat_types_next;
    stat_types_next[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types_next[static_cast<size_t>(
        get_stat_type_for_pool(*(next_block->pool)))] = true;
    update_stat_array(stats_next.allocation, 1, stat_types_next);
    update_stat_array(
        stats_next.allocated_bytes, next_block->size, stat_types_next);
    update_stat_array(stats_next.active, 1, stat_types_next);
    update_stat_array(
        stats_next.active_bytes, next_block->size, stat_types_next);
  }
};

void THNCachingAllocator::malloc(void** devPtr, size_t size, aclrtStream stream, int device) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  if (device == -1) {
    C10_NPU_CHECK(aclrtGetDevice(&device));
  }
  // process outstanding npuEvents
  process_events();
  size = round_size(size);
  DeviceStats& stats = get_stats_for_device(device);

  Block search_key(device, stream, size);
  auto& pool = get_pool(size);

  DeviceStats_& stats_ = get_stats_for_device_(device);
  StatTypes stat_types;
  stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
  stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

  auto find_free_block = [&]() -> Block* {
    auto it = pool.lower_bound(&search_key);
    if (it != pool.end() && (*it)->device == device &&
        (*it)->stream == stream) {
      Block* block = *it;
      pool.erase(it);
      return block;
    }
    return nullptr;
  };

  Block* block = find_free_block();
  if (block == nullptr) {
    bool freed_memory = false;
    for (const auto& name : FreeNPUMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeNPUMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    if (freed_memory) {
      block = find_free_block();
    }
  }
  if (block == nullptr) {
    void* ptr = nullptr;
    size_t alloc_size = get_allocation_size(size);
    aclError err = npu_malloc_retry(device, &ptr, alloc_size);

    if (err != ACL_ERROR_NONE) {
      if (err == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        size_t device_free;
        size_t device_total;
        C10_NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));

        const auto& stats = get_stats_for_device(device);

        stats_.num_ooms += 1;
        // "total capacity": total global memory on NPU
        // "already allocated": memory allocated by the program using the
        //                      caching allocator
        // "free": free memory as reported by the NPU API
        // "cached": memory held by the allocator but not used by the program
        //
        // The "allocated" amount  does not include memory allocated outside
        // of the caching allocator, such as memory allocated by other
        // programs or memory held by the driver.
        //
        // The sum of "allocated" + "free" + "cached" may be less than the
        // total capacity due to memory held by the driver and usage by other
        // programs.
        //
        // Note that at this point npu_malloc_retry has already returned all
        // possible "cached" memory to the driver. The only remaining "cached"
        // memory is split from a larger block that is partially in-use.
        AT_ERROR(
            "NPU out of memory. Tried to allocate ",
            format_size(alloc_size),
            " (NPU ",
            device,
            "; ",
            format_size(device_total),
            " total capacity; ",
            format_size(stats.amount_allocated),
            " already allocated; ",
            format_size(device_free),
            " free; ",
            format_size(stats.amount_cached - stats.amount_allocated),
            " cached)");
      } else {
        C10_NPU_CHECK(err);
      }
    }
    stats.increaseCached(alloc_size);
    block = new Block(device, stream, alloc_size, &pool, ptr);

    update_stat_array(stats_.segment, 1, stat_types);
    update_stat_array(stats_.reserved_bytes, alloc_size, stat_types);
  }

  Block* remaining = nullptr;
  AT_ASSERT(block);

  const bool already_split = block->is_split();
  if (should_split(block, size)) {
    remaining = block;
    block = new Block(device, stream, size, &pool, block->ptr);
    block->prev = remaining->prev;
    if (block->prev) {
      block->prev->next = block;
    }
    block->next = remaining;

    remaining->prev = block;
    remaining->ptr = static_cast<char*>(remaining->ptr) + size;
    remaining->size -= size;
    pool.insert(remaining);

    if (already_split) {
      // An already-split inactive block is being shrunk by size bytes.
      update_stat_array(
          stats_.inactive_split_bytes, -block->size, stat_types);
    } else {
      // A new split inactive block is being created from a previously unsplit
      // block, size remaining->size bytes.
      update_stat_array(
          stats_.inactive_split_bytes, remaining->size, stat_types);
      update_stat_array(stats_.inactive_split, 1, stat_types);
    }
  } else if (already_split) {
    // An already-split block is becoming active
    update_stat_array(stats_.inactive_split_bytes, -block->size, stat_types);
    update_stat_array(stats_.inactive_split, -1, stat_types);
  }

  block->allocated = true;
  allocated_blocks[block->ptr] = block;
  *devPtr = block->ptr;
  stats.increaseAllocated(block->size);

  c10::reportMemoryUsageToProfiler(
      block, block->size, c10::Device(at_npu::key::NativeDeviceType, device));

  update_stat_array(stats_.allocation, 1, stat_types);
  update_stat_array(stats_.allocated_bytes, block->size, stat_types);
  update_stat_array(stats_.active, 1, stat_types);
  update_stat_array(stats_.active_bytes, block->size, stat_types);
}

THNCachingAllocator caching_allocator;

static void NPUCachingDeleter(void* ptr) {
  caching_allocator.free(ptr);
}

// NB: I decided not to fold this into THNCachingAllocator, because the latter
// has a lot more methods and it wasn't altogether clear that they should
// actually be publically exposed
struct NPUCachingAllocator : public c10::Allocator {
  c10::DataPtr allocate(size_t size) const override {
    int device = 0;
    C10_NPU_CHECK(aclrtGetDevice(&device));
    void* r = nullptr;
    if (size != 0) {
      caching_allocator.malloc(
          &r, size, c10_npu::getCurrentNPUStreamNoWait(device), device);
    }
    return {r, r, &NPUCachingDeleter, c10::Device(at_npu::key::NativeDeviceType, device)};
  }
  c10::DeleterFnPtr raw_deleter() const override {
    return &NPUCachingDeleter;
  }
};

std::tuple<c10::DataPtr, c10::DataPtr> allocate_adjacent(size_t size1, size_t size2) {
  int device = 0;
  C10_NPU_CHECK(aclrtGetDevice(&device));
  void* ptr_pre = nullptr;
  void* ptr_next = nullptr;
  caching_allocator.allocate_adjacent_ptr(
      size1,
      size2,
      &ptr_pre,
      &ptr_next,
      c10_npu::getCurrentNPUStreamNoWait(device));

  c10::DataPtr data_pre = {
      ptr_pre, ptr_pre, &NPUCachingDeleter, c10::Device(at_npu::key::NativeDeviceType, device)};
  c10::DataPtr data_next = {
      ptr_next, ptr_next, &NPUCachingDeleter, c10::Device(at_npu::key::NativeDeviceType, device)};
  std::tuple<c10::DataPtr, c10::DataPtr> adjacent_dataptr =
      std::make_tuple(std::move(data_pre), std::move(data_next));

  return adjacent_dataptr;
}

NPUCachingAllocator device_allocator;

c10::Allocator* get(void) {
  return &device_allocator;
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.cacheInfo(dev_id, cachedAndFree, largestBlock);
}

void* getBaseAllocation(void* ptr, size_t* size) {
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) {
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex() {
  return &caching_allocator.npu_free_mutex;
}

static inline void assertValidDevice(int device) {
  int device_num = c10_npu::device_count();
  AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

DeviceStats_ getDeviceStats(int device) {
  assertValidDevice(device);
  return caching_allocator.getStatsForDevice(device);
}

void resetAccumulatedStats(int device) {
  assertValidDevice(device);
  caching_allocator.resetAccumulatedStats(device);
}

void resetPeakStats(int device) {
  assertValidDevice(device);
  caching_allocator.resetPeakStats(device);
}

std::vector<SegmentInfo> snapshot() {
  return caching_allocator.snapshot();
}

void NpuAllocatorInsertRecordedEvent(aclrtEvent event) {
  return caching_allocator.insertRecordedEvent(event);
}

uint64_t currentMemoryAllocated(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).amount_allocated;
}

uint64_t maxMemoryAllocated(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).max_amount_allocated;
}

void resetMaxMemoryAllocated(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.get_stats_for_device(device);
  stats.max_amount_allocated = stats.amount_allocated;
}

uint64_t currentMemoryCached(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).amount_cached;
}

uint64_t maxMemoryCached(int device) {
  assertValidDevice(device);
  return caching_allocator.get_stats_for_device(device).max_amount_cached;
}

void resetMaxMemoryCached(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.get_stats_for_device(device);
  stats.max_amount_cached = stats.amount_cached;
}

//
// In NPU IPC, sender sends a tensor to receiver, getIpcDevPtr
// is called by the receiving process to map the NPU memory from the sending
// process into its own address space.
//
// NPU IPC only allows sharing a big memory block associated with a
// npuIpcMemHandle_t and it can be opened only **once** per context per
// process. There can be multiple types of storage in the same IPC mem block, so
// we must cache the device ptr to construct typed storage as it comes.
//
// ipcMemHandle_to_devptr maps a npuIpcMemHandle_t to a device pointer in the
// process that can be used to access the memory block in the sender process. It
// only saves a weak_ptr of the device pointer in the map, the shared_ptr will
// be used to reconstruct all storages in this NPUMalloc allocation. And it
// will deleted in npuIpcCloseMemHandle when its reference count is 0.
//
namespace {
std::mutex IpcMutex;
std::unordered_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
} // namespace


void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device = 0;
  C10_NPU_CHECK(aclrtGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, nbytes, c10_npu::getCurrentNPUStreamNoWait(device), device);
  return r;
}

void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream) {
  if (nbytes == 0) {
    return nullptr;
  }
  void* r = nullptr;
  caching_allocator.malloc(&r, nbytes, stream);
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

void FreeDeviceCachedMemory(int device)
{
  caching_allocator.free_cached_blocks(device);
}

} // namespace NPUCachingAllocator
} // namespace c10_npu
