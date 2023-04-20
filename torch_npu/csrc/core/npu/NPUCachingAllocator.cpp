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
#include <regex>
#include <set>
#include <vector>

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/UniqueVoidPtr.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

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
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
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
using stream_set = ska::flat_hash_set<c10_npu::NPUStream>;

constexpr size_t kMinBlockSize = 512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocs to 2 MiB

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

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

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        update_stat(stat_array[stat_type], amount);
      });
}

struct Block;
using Comparison = bool (*)(const Block*, const Block*);

struct BlockPool{
  std::set<Block*, Comparison> blocks;
  const bool is_small;

  BlockPool(Comparison comparator, bool small)
      : blocks(comparator),
        is_small(small) {}
};

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

struct AllocParams {
  AllocParams(int device, size_t size, aclrtStream stream, BlockPool* pool, size_t alloc_size,
              DeviceStats& stats) :
    search_key(device, stream, size),
    pool(pool),
    alloc_size(alloc_size),
    block(nullptr),
    err(ACL_ERROR_NONE) {}

  int device() const { return search_key.device; }
  aclrtStream stream() const { return search_key.stream; }
  size_t size() const { return search_key.size; }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
  aclError err;
};

} // namespace

class CachingAllocatorConfig {

 public:

  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

 private:

  size_t m_max_split_size;

  static CachingAllocatorConfig &instance() {
    static CachingAllocatorConfig *s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      inst->parseArgs();
      return inst;
    })();
    return *s_instance;
  }

  CachingAllocatorConfig() : m_max_split_size(std::numeric_limits<size_t>::max()) {}

  void parseArgs() {
      const char *val = getenv("PYTORCH_NPU_ALLOC_CONF");
      if (val != nullptr) {
        const std::string config(val);
        std::regex exp("[\\s,]+");
        std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
        std::sregex_token_iterator end;
        std::vector<std::string> options(it, end);

        for (auto option : options) {
          std::regex exp2("[:]+");
          std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
          std::sregex_token_iterator end2;
          std::vector<std::string> kv(it2, end2);
          if (kv.size() >= 2) {
            /* Maximum split size in MB.  Limited to large size blocks */
            if (kv[0].compare("max_split_size_mb") == 0) {
              size_t val2 = stoi(kv[1]);
              TORCH_CHECK(val2 > kLargeBuffer / (1024 * 1024),
                  "CachingAllocator option max_split_size_mb too small, must be > ",
                  kLargeBuffer / (1024 * 1024),
                  "");
              val2 = std::max(val2, kLargeBuffer / (1024 * 1024));
              val2 = std::min(val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
              m_max_split_size = val2 * 1024 * 1024;
            } else {
              TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", kv[0]);
            }
          }
        }
      }
    }
};

class DeviceCachingAllocator {

 private:

  // lock around all operations
  mutable std::recursive_mutex mutex;
  
  // lock around taskqueue events operations
  mutable std::mutex recorded_event_mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream
  ska::flat_hash_set<Block*> active_blocks;

  // outstanding acl events
  std::deque<std::pair<aclrtEvent, Block*>> npu_events;
  
  // outstanding taskqueue events
  std::set<aclrtEvent> recorded_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  // record maximum allowed memory.
  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;

 public:

  DeviceCachingAllocator() :
    large_blocks(BlockComparator, false),
    small_blocks(BlockComparator, true) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(int device, size_t size, aclrtStream stream) {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (device == -1) {
        C10_NPU_CHECK(aclrtGetDevice(&device));
    }

    // process outstanding npuEvents
    process_events();
    size = round_size(size);
    auto& pool = get_pool(size);

    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    bool block_found =
      // Search pool
      get_free_block(params)
      // Trigger callbacks and retry search
      || (trigger_free_memory_callbacks(params) && get_free_block(params))
      // Attempt allocate
      || alloc_block(params, false)
      // Free enough available cached blocks to satisfy alloc and retry alloc.
      || (release_available_cached_blocks(params) && alloc_block(params, false))
      // Free all non-split cached blocks and retry alloc.
      || (release_cached_blocks() && alloc_block(params, true));

    if (!block_found) {
      if (params.err == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        size_t device_free;
        size_t device_total;
        C10_NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
        
        std::string allowed_info;
        if (set_fraction) {
          allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
        }
        stats.num_ooms += 1;
        // "total capacity": total global memory on NPU
        // "allowed": memory is allowed to use, which set by fraction.
        // "already allocated": memory allocated by the program using the
        //                      caching allocator
        // "free": free memory as reported by the NPU API
        // "cached": memory held by the allocator but not used by the program
        //
        // The "allocated" amount  does not include memory allocated outside
        // of the caching allocator, such as memory allocated by other programs
        // or memory held by the driver.
        //
        // The sum of "allocated" + "free" + "cached" may be less than the
        // total capacity due to memory held by the driver and usage by other
        // programs.
        //
        // Note that at this point free_cached_blocks has already returned all
        // possible "cached" memory to the driver. The only remaining "cached"
        // memory is split from a larger block that is partially in-use.
        AT_ERROR(
            "NPU out of memory. Tried to allocate ",
            format_size(alloc_size),
            " (NPU ", device, "; ",
            format_size(device_total),
            " total capacity; ",
            format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
            " already allocated; ",
            format_size(device_free),
            " free; ",
            allowed_info,
            format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
            " reserved in total by PyTorch)",
            " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.");
      } else {
        C10_NPU_CHECK(params.err);
      }
    }
    Block* block = params.block;
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
      pool.blocks.insert(remaining);

      if (already_split) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
      } else {
        // A new split inactive block is being created from a previously unsplit block,
        // size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(stats.inactive_split_bytes[stat_type], remaining->size);
          update_stat(stats.inactive_split[stat_type], 1);
        });
      }
    } else if (already_split) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats.inactive_split_bytes[stat_type], -block->size);
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    block->allocated = true;
    active_blocks.insert(block);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(stats.allocated_bytes[stat_type], block->size);
      update_stat(stats.active[stat_type], 1);
      update_stat(stats.active_bytes[stat_type], block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    ASCEND_LOGD("PTA CachingAllocator malloc: malloc = %zu, address = %lu, cached = %lu, allocated = %lu",
        size,
        block->ptr,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);

    return block;
  }

  void free(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(stats.allocated_bytes[stat_type], -block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }

    ASCEND_LOGD("PTA CachingAllocator free: free = %zu, address = %lu, cached = %lu, allocated = %lu",
        orig_block_size,
        orig_block_ptr,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);

  }

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
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

  void recordStream(Block* block, c10_npu::NPUStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free;
    size_t device_total;
    C10_NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks();
  }

  /** Retrieves info (total size + largest block) of the memory cache **/
  void cacheInfo(size_t* total, size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cache_info_aux(large_blocks, total, largest);
    cache_info_aux(small_blocks, total, largest);
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
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
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
    }

    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. **/
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
      segment_info.is_large = (!head_block->pool->is_small);

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

    std::sort(result.begin(), result.end(), [](const SegmentInfo& a, const SegmentInfo& b) {
      return a.address < b.address;
    });

    return result;
  }

  static size_t round_size(size_t size) {
    size = size + 32;
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  void insertRecordedEvent(aclrtEvent event) {
    std::lock_guard<std::mutex> lock(recorded_event_mutex);
    recorded_events.insert(event);
  }

 private:

  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  /** moves a block into a pool of cached free blocks **/
  void free_block(Block* block) {
    AT_ASSERT(!block->allocated && block->event_count == 0);

    size_t original_block_size = block->size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    pool.blocks.insert(block);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.inactive_split[stat_type], net_change_inactive_split_blocks);
      update_stat(stats.inactive_split_bytes[stat_type], net_change_inactive_split_size);
      update_stat(stats.active[stat_type], -1);
      update_stat(stats.active_bytes[stat_type], -original_block_size);
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block, or 0 on failure. **/
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

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
    dst->size += subsumed_size;
    pool.blocks.erase(src);
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
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) && (remaining > kSmallSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
        return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) && ((*it)->size >= p.size() + kLargeBuffer))
        return false;
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeNPUMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
        FreeNPUMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  bool alloc_block(AllocParams& p, bool isRetry) {
    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }
    
    if (set_fraction && total_allocated_memory + size > allowed_memory_maximum) {
      p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
    } else {
      p.err = aclrtMalloc(
          &ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    }
  
    if (p.err != ACL_ERROR_NONE) {
      return false;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_segments, 1);
    ASCEND_LOGD("pta_memory acl_malloc: malloc = %zu, ret = %d", size, p.err);

    return (p.block != nullptr);
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() == std::numeric_limits<size_t>::max()) {
      return false;
    }
    BlockPool &pool = *p.pool;
    Block key = p.search_key;
    key.size =
        (key.size < CachingAllocatorConfig::max_split_size()) ? CachingAllocatorConfig::max_split_size() : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks, starting with the largest
      if (it == pool.blocks.begin()){
        return false;
      }
      size_t totalReleased = 0;
      // Back up one item.  Now on the largest block for the correct stream
      --it;  
      while ((totalReleased < key.size) && ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
            ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur);
        } else {
          release_block(*cur);
          break;
        }
      }
      if (totalReleased < key.size) {
        return false;
      }
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks() {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();

    // Free all non-split cached blocks
    c10_npu::npuSynchronizeDevice();
    release_blocks(large_blocks);
    release_blocks(small_blocks);

    return true;
  }

  void release_block(Block* block) {
    aclrtFree((void*)block->ptr);
    total_allocated_memory -= block->size;

    auto* pool = block->pool; 
    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(stats.reserved_bytes[stat_type], -block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);

    pool->blocks.erase(block);
    delete block;
    }

  void release_blocks(BlockPool& pool) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block *block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  void free_blocks(BlockPool& blocks) {
    // Frees all non-split blocks
    auto it = blocks.blocks.begin();
    while (it != blocks.blocks.end()) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        aclrtFree((void*)block->ptr);
        total_allocated_memory -= block->size;

        StatTypes stat_types = {false};
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
        update_stat_array(stats.segment, -1, stat_types);
        update_stat_array(stats.reserved_bytes, -block->size, stat_types);
        ASCEND_LOGD("pta_memory acl_free: free_size = %zu", block->size);
        auto cur = it;
        ++it;
        blocks.blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
  }

  aclrtEvent create_event_internal() {
    aclrtEvent event = nullptr;
    C10_NPU_CHECK(c10_npu::acl::AclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE));
    return event;
  }

  void free_event_internal(aclrtEvent event) {
    C10_NPU_CHECK(aclrtDestroyEvent(event));
  }

  void synchronize_and_free_events() {
    // Synchronize on outstanding events and then free associated blocks.

    for (auto& e : npu_events) {
      aclrtEvent event = e.first;
      {
        std::lock_guard<std::mutex> lock(recorded_event_mutex);
        auto it = recorded_events.find(event);
        if (c10_npu::option::OptionsManager::CheckQueueEnable() && it == recorded_events.end()) {
            break;
        }
      }
      Block* block = e.second;

      C10_NPU_CHECK(aclrtSynchronizeEvent(event));
      {
        std::lock_guard<std::mutex> lock(recorded_event_mutex);
        auto it = recorded_events.find(event);
        if (it != recorded_events.end()) {
          recorded_events.erase(it);
        }
      }
      free_event_internal(event);

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    npu_events.clear();
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

      aclrtEvent event = create_event_internal();
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
        if (c10_npu::option::OptionsManager::CheckQueueEnable() && it == recorded_events.end()) {
            break;
        }
      }

      c10_npu::acl::aclrtEventRecordedStatus status = c10_npu::acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
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
      
      free_event_internal(event);

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      npu_events.pop_front();
    }
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cache_info_aux(BlockPool& blocks, size_t* total, size_t* largest) {
    for (auto it = blocks.blocks.begin(); it != blocks.blocks.end(); ++it) {
      size_t blocksize = (*it)->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }
};

class THNCachingAllocator {

 private:

  std::mutex mutex;

  // allocated blocks by device pointer
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  // lock around calls to aclFree (to prevent deadlocks with HCCL)
  mutable std::mutex npu_free_mutex;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:

  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  std::mutex* getNpuFreeMutex() const {
    return &npu_free_mutex;
  }

  Block* get_allocated_block(void* ptr, bool remove=false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void init(int device_count) {
    int size = device_allocator.size();
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, aclrtStream stream) {
    Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = static_cast<void*>(block->ptr);
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    device_allocator[block->device]->free(block);
  }

  void setMemoryFraction(double fraction, int device) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && device < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction  && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    int activated_device;
    aclrtGetDevice(&activated_device);
    if (activated_device != device) {
        aclrtSetDevice(device);
    }
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void emptyCache() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++)
      device_allocator[i]->emptyCache();
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
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
    if (ptr.get_deleter() != &raw_delete)
      return;

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    device_allocator[block->device]->recordStream(block, stream);
  }

  std::vector<SegmentInfo> snapshot() {
    std::vector<SegmentInfo> result;
    int count = device_allocator.size();
    for (int i = 0; i < count; i++) {
      auto snap = device_allocator[i]->snapshot();
      result.insert(result.end(), snap.begin(), snap.end());
    }

    return result;
  }
};

THNCachingAllocator caching_allocator;

static void NPUCachingDeleter(void* ptr) {
  caching_allocator.free(ptr);
}

// NB: I decided not to fold this into THNCachingAllocator, because the latter
// has a lot more methods and it wasn't altogether clear that they should
// actually be publically exposed
struct NpuCachingAllocator : public c10::Allocator {
  c10::DataPtr allocate(size_t size) const override {
    int device = 0;
    C10_NPU_CHECK(aclrtGetDevice(&device));
    void* r = nullptr;
    if (size != 0) {
      caching_allocator.malloc(&r, device, size, c10_npu::getCurrentNPUStreamNoWait(device));
    }
    return {r, r, &NPUCachingDeleter, c10::Device(at_npu::key::NativeDeviceType, device)};
  }
  c10::DeleterFnPtr raw_deleter() const override {
    return &NPUCachingDeleter;
  }
};

NpuCachingAllocator device_allocator;

c10::Allocator* get(void) {
  return &device_allocator;
}

void init() {
  uint32_t device_count = 0;
  C10_NPU_CHECK(aclrtGetDeviceCount(&device_count));
  caching_allocator.init(device_count);
}

void setMemoryFraction(double fraction, int device) {
  caching_allocator.setMemoryFraction(fraction, device);
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.device_allocator[dev_id]->cacheInfo(cachedAndFree, largestBlock);
}

void* getBaseAllocation(void* ptr, size_t* size) {
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) {
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex() {
  return caching_allocator.getNpuFreeMutex();
}

static inline void assertValidDevice(int device) {
  int device_num = c10_npu::device_count();
  AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

DeviceStats getDeviceStats(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->getStats();
}

void resetAccumulatedStats(int device) {
  assertValidDevice(device);
  caching_allocator.device_allocator[device]->resetAccumulatedStats();
}

void resetPeakStats(int device) {
  assertValidDevice(device);
  caching_allocator.device_allocator[device]->resetPeakStats();
}

std::vector<SegmentInfo> snapshot() {
  return caching_allocator.snapshot();
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device = 0;
  C10_NPU_CHECK(aclrtGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, device, nbytes, c10_npu::getCurrentNPUStreamNoWait(device));
  return r;
}

void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_NPU_CHECK(aclrtGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, device, nbytes, stream);
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

void FreeDeviceCachedMemory(int device) {
  caching_allocator.device_allocator[device]->emptyCache();
}

void NpuAllocatorInsertRecordedEvent(aclrtEvent event) {
  int device = 0;
  C10_NPU_CHECK(aclrtGetDevice(&device));
  return caching_allocator.device_allocator[device]->insertRecordedEvent(event);
}

} // namespace NPUCachingAllocator
} // namespace c10_npu
