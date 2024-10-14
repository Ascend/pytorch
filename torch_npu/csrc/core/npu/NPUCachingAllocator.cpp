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
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/UniqueVoidPtr.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/core/npu/NPURecovery.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "NPUBlockHandle.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"

std::string format_size(uint64_t size)
{
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
constexpr size_t kAlignRoundLarge = 16384; // round up large allocs to 16 KB

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
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

struct BlockPool{
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  const bool is_small;

  BlockPool(bool small)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small) {}
};

struct ExpandableSegment;

struct Block {
    int device; // npu
    aclrtStream stream; // allocation stream
    stream_set stream_uses; // streams on which the block was used
    size_t size; // block size in bytes
    size_t requested_size; // memory originally requested
    BlockPool* pool; // owning memory pool
    void* ptr; // memory address
    bool allocated; // in-use flag
    bool mapped{true}; // is the virtual address range this Block references
                       // backed by physical pages. Always true when
                      // expandable_segment_ is null. When false
                      // This Block will be aligned to the segment size
                      // of its expandable_segment_.
    Block* prev; // prev block if split from a larger allocation
    Block* next; // next block if split from a larger allocation
    int event_count; // number of outstanding NPU events
    int gc_count{0}; // counter for prioritizing older / less useful blocks for
                     // garbage collection
    ExpandableSegment* expandable_segment_{nullptr};
    bool is_safe{true};
    std::shared_ptr<c10::GatheredContext> context_when_allocated;
    // only set for the first block in the segment (when prev == null)
    // this records the frame information when cudaMalloc was called
    // whereas context_when_allocated records the last time we handed this
    // memory out from our cache.
    std::shared_ptr<c10::GatheredContext> context_when_segment_allocated;

    Block(int device, aclrtStream stream, size_t size, BlockPool* pool, void* ptr)
        : device(device),
          stream(stream),
          stream_uses(),
          size(size),
          requested_size(0),
          pool(pool),
          ptr(ptr),
          allocated(0),
          prev(nullptr),
          next(nullptr),
          event_count(0),
          gc_count(0) {}

    // constructor for search key
    Block(int device, aclrtStream stream, size_t size)
        : device(device),
          stream(stream),
          stream_uses(),
          size(size),
          requested_size(0),
          pool(nullptr),
          ptr(nullptr),
          allocated(0),
          prev(nullptr),
          next(nullptr),
          event_count(0),
          gc_count(0) {}

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }

    void splice(Block* before, Block* after)
    {
        if (before) {
            TORCH_INTERNAL_ASSERT(before->next == after, PTA_ERROR(ErrCode::PTR));
            before->next = this;
        }
        prev = before;
        if (after) {
            TORCH_INTERNAL_ASSERT(after->prev == before, PTA_ERROR(ErrCode::PTR));
            after->prev = this;
        }
        next = after;
    }
};

struct SegmentRange {
  char* ptr;
  size_t size;
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};


/*
Note [Expandable Segments]
Rationale
For large (>2MB) allocations, the allocator calls aclrtMalloc to get allocations
that are the same size as whataclrtMalloc the user requests. In the future, parts of these
allocations can be reused for other requests if they are free. This works well
when the program makes many requests of exactly the same size or of sizes that
even multiples of that size. Many deep learning models follow this behavior.
However, one common exception is when the batch size changes slightly from one
iteration to the next, e.g. in batched inference. When the program runs
initially with batch size N, it will make allocations appropriate for that size.
If in the future, it runs at size N - 1, the existing allocations will still be
big enough. However, if it runs at size N + 1, then it will have to make new
allocations that are slightly larger. Not all the tensors are the same size.
Some might be (N + 1)*A and others (N + 1)*A*B where A and B are some non-batch
dimensions in the model. Because the allocator reuses existing allocations when
they are big enough, some number of (N + 1)*A allocations will actually fit in
the already existing N*B*A segments, though not perfectly. As the model runs it
will partially fill up all of these segments leaving unusable free slices of
memory at the end of these segments. The allocator at some point will need to
aclrtMalloc a new (N + 1)*A*B segment. If there is not enough memory, there is
now no way to recover the slices of memory that are free at the end of existing
segments. With models 50+ layers deep, this pattern might repeat 50+ times
creating many slivers.
Approach
Expandable segments allows the allocator to create a segment initially and then
expand its size later when more memory is needed. Instead of making one segment
per allocation, it tries to make one segment (per stream) that grows as
necessary. Now when the N + 1 case runs, the allocations will tile nicely into
the one large segment until it fills up. Then more memory is requested and
appended to the end of the segment. This process does not create as many slivers
of unusable memory, so it is more likely to succeed at finding this memory.
Implementation
The expandable_segments:True option is used to enable/disable this behavior. We
use npu's low-level memory APIs, which are similar to mmap, to extend the
memory segments. These APIs separate the allocation of physical memory
(AclrtMallocPhysical) from the allocation of virtual address space (AclrtReserveMemAddress)
and the associate between them AclrtMapMem.
When we allocate a new segment, we allocate enough address space to map
basically the entire physical memory of the NPU (there is 256TiB of address
space), but we only map enough physical memory to handle the current amount of
memory needed by the program. As more is requested, we add more physical memory
to the segment. This can work at the granularity of NPU pages which are 2MiB
currently.
If we end up out of memory, we can unmap all the memory in our segment
corresponding to empty physical pages, and return it to NPU for use at another
address in the segment or in a segment for a different stream.
A current limitation of NPU's API is that physical memory
(aclrtDrvMemHandle) cannot be split up after it is mapped even if the
handle holds multiple NPU pages. The cost to map/unmap memory is proportional to
the number of physical memory chunks that were allocated (mapping 10 separately
allocated 2MiB pages takes 10x time compared to mapping one 20MiB physical
allocation of 10 pages).  Changing memory mappings also appears to involve at
least some synchronous actions with the NPU and so should be considered an
expensive operation. To limit overhead, we use 2MiB pages for our small pool and
20MiB pages for our large pool. Initially allocation using expandable_blocks
will be slower than aclrtMalloc, though still in the milliseconds range for
mapping the entire memory.
When mapping new memory to expand the segment, we look for the lowest address at
which we can fit a new allocation by adding new pages. Normally this will be at
the end of the block. But if have previously unmapped blocks earlier in the
segment during an OOM, it will first try to fill in those gaps to keep the
segment as a single block. By allocating at the lowest address we encourage
the split up parts of the block to merge into a single block again, reducing
fragmentation potential.
Allocation of blocks in the segment uses the same best-fit heuristics of the
rest of the allocator.
Expandable blocks can be enabled/disabled throughout the run of a program. When
disabled, the allocator will not put new allocations in an expandable block.
Limitations
* Slightly slower initial memory allocation speed.
* IPC of npu tensors (e.g. for multiprocess dataloaders) is not supported.
However, it is possible to temporarily disable (expandable_segments:False) the
bevhavior for allocator tensors that need to be used cross-process.
*/

struct ExpandableSegment {
  ExpandableSegment(
      int device,
      aclrtStream stream,
      size_t size)
      : device_(device),
        stream_(stream),
        max_handles_(0),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size) {
    size_t device_free;
    size_t device_total;
    NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
    // we allocate enough address space for 1 1/8 the total memory on the NPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(device_total + device_total / 8);
    NPU_CHECK_ERROR(c10_npu::acl::AclrtReserveMemAddress(
        &ptr_, segment_size_ * max_handles_, 0, NULL, 1));
    ASCEND_LOGD(
        "NPUCachingAllocator malloc by AclrtReserveMemAddress: size=%zu",
        segment_size_ * max_handles_);
  }
  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr, PTA_ERROR(ErrCode::PTR));
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }
    while (end > handles_.size()) {
      handles_.emplace_back(c10::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
        TORCH_INTERNAL_ASSERT(!handles_.at(i), PTA_ERROR(ErrCode::VALUE));
        aclrtDrvMemHandle handle = nullptr;
        aclrtPhysicalMemProp prop = {};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
        prop.reserve = 0;
        auto status =
            c10_npu::acl::AclrtMallocPhysical(&handle, segment_size_, &prop, 0);
        if (status == ACL_ERROR_RT_MEMORY_ALLOCATION) {
            for (auto j : c10::irange(begin, i)) {
                auto h = handles_.at(j).value();
                handles_.at(j) = c10::nullopt;
                NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h));
            }
            trimHandles();
            return rangeFromHandles(begin, begin);
        }
        NPU_CHECK_ERROR(status, "aclrtMallocPhysical");
        handles_.at(i) = handle;
    }
    for (auto i : c10::irange(begin, end)) {
      NPU_CHECK_ERROR(c10_npu::acl::AclrtMapMem(
          ptr_ + i * segment_size_,
          segment_size_,
          0,
          handles_.at(i).value(),
          0));
    }
    ASCEND_LOGD(
        "NPUCachingAllocator map: segment_size=%zu", segment_size_);
    return rangeFromHandles(begin, end);
  }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char* ptr() const {
    return (char*)ptr_;
  }

  size_t size() const {
    return max_handles_ * segment_size_;
  }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr_));
    ASCEND_LOGD("NPUCachingAllocator free by AclrtReleaseMemAddress");
  }

 private:
  void unmapHandles(size_t begin, size_t end) {
    // note: unlike aclrtFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::npu::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    NPU_CHECK_ERROR(aclrtSynchronizeStream(stream_));
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuStreamSynchronization(
            reinterpret_cast<uintptr_t>(stream_));
    }
#endif
    for (auto i : c10::irange(begin, end)) {
      aclrtDrvMemHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      NPU_CHECK_ERROR(c10_npu::acl::AclrtUnmapMem(ptr_ + segment_size_ * i));
      NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h));
    }
      ASCEND_LOGD("NPUCachingAllocator unmap: segment_size=%zu", segment_size_);
    trimHandles();
  }

  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }

  void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
    auto start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }

  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }

  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }

  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }

  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }

  int device_;
  aclrtStream stream_;
  void* ptr_{};
  size_t max_handles_;
  size_t segment_size_;
  std::vector<c10::optional<aclrtDrvMemHandle>> handles_;
};

static bool BlockComparatorSize(const Block* a, const Block* b) {
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

static bool BlockComparatorAddress(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
        reinterpret_cast<uintptr_t>(b->stream);
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}

struct AllocParams {
  AllocParams(
      int device,
      size_t size,
      aclrtStream stream,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, stream, size),
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

class EventPool {
public:
  using Event = std::unique_ptr<c10_npu::NPUEvent, std::function<void(c10_npu::NPUEvent*)>>;
  // Explicit device count
  EventPool() : pools_(c10_npu::device_count()) {}

  Event get(int device) {
    TORCH_INTERNAL_ASSERT(0 <= device, PTA_ERROR(ErrCode::VALUE));
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()), PTA_ERROR(ErrCode::VALUE));
    auto& pool = pools_[device];
    auto destructor = [&pool](c10_npu::NPUEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<c10_npu::NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<c10_npu::NPUEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

} // namespace

class CachingAllocatorConfig {
 public:

  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
    return instance().m_expandable_segments;
  }

  static size_t base_addr_aligned_size()
  {
      return instance().m_base_addr_aligned_size;
  }

  static CachingAllocatorConfig &instance() {
    static CachingAllocatorConfig *s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      const char* env = getenv("PYTORCH_NPU_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:

  size_t m_max_split_size;
  double m_garbage_collection_threshold;
  bool m_expandable_segments;
  bool set_expandable_segments_flag = false;
  size_t m_base_addr_aligned_size = kAlignRoundLarge;

  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0),
        m_expandable_segments(true),
        m_base_addr_aligned_size(kAlignRoundLarge)
        {
            void* ptr = nullptr;
            auto status = c10_npu::acl::AclrtReserveMemAddress(&ptr, 512, 0, NULL, 1);
            if (status == ACL_ERROR_NONE) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr));
            } else {
                TORCH_NPU_WARN_ONCE("expandable_segments feature is not supportted \
                    and the possible cause is that driver and firmware packages do not match.");
                m_expandable_segments = false;
            }
        }

  void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseExpandableSegments(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseAddrAlignSize(
      const std::vector<std::string>& config,
      size_t i);
};

void CachingAllocatorConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void CachingAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i].compare(std::string(1, c)) == 0,
      "Error parsing CachingAllocator settings, expected ", c, PTA_ERROR(ErrCode::PARAM));
}

size_t CachingAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val1 = static_cast<size_t>(stoi(config[i]));
    TORCH_CHECK(
        val1 > kLargeBuffer / (1024 * 1024),
        "CachingAllocator option max_split_size_mb too small, must be > ",
        kLargeBuffer / (1024 * 1024), PTA_ERROR(ErrCode::VALUE));
    val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", PTA_ERROR(ErrCode::PARAM));
  }
  return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(
        val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", PTA_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", PTA_ERROR(ErrCode::VALUE));
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(
        false, "Error, expecting garbage_collection_threshold value", PTA_ERROR(ErrCode::VALUE));
  }
  return i;
}

size_t CachingAllocatorConfig::parseExpandableSegments(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        i < config.size() && (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for expandable_segments", PTA_ERROR(ErrCode::PARAM));
    m_expandable_segments = (config[i] == "True");
    if (m_expandable_segments) {
        void* ptr = nullptr;
        auto status = c10_npu::acl::AclrtReserveMemAddress(&ptr, 512, 0, NULL, 1);
        if (status == ACL_ERROR_NONE) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr));
        } else {
            NPU_CHECK_SUPPORTED_OR_ERROR(status, "aclrtReserveMemAddress");
            TORCH_NPU_WARN_ONCE("expandable_segments setting failure, now change to `False`.");
            m_expandable_segments = false;
        }
    }
  } else {
    TORCH_CHECK(
        false, "Error, expecting expandable_segments value", PTA_ERROR(ErrCode::PARAM));
  }
  return i;
}

size_t CachingAllocatorConfig::parseAddrAlignSize(
    const std::vector<std::string>& config,
    size_t i)
{
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val = static_cast<size_t>(stoi(config[i]));
        TORCH_CHECK(config[i].length() == std::to_string(val).length(),
                    "CachingAllocator option base_addr_aligned_kb error, must be [0~16], dtype is int",
                    OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(val >= 0, "CachingAllocator option base_addr_aligned_kb error, must be [0~16], dtype is int",
                    OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(val <= kAlignRoundLarge / 1024,
                    "CachingAllocator option base_addr_aligned_kb error, must be [0~16], dtype is int",
                    OPS_ERROR(ErrCode::VALUE));
        m_base_addr_aligned_size = val * 1024;
    } else {
        TORCH_CHECK(false, "Error, expecting base_addr_aligned_kb value", OPS_ERROR(ErrCode::VALUE));
    }
    return i;
}

void CachingAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_garbage_collection_threshold = 0;

  if (env == nullptr) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    if (config[i].compare("max_split_size_mb") == 0) {
      i = parseMaxSplitSize(config, i);
    } else if (config[i].compare("garbage_collection_threshold") == 0) {
      i = parseGarbageCollectionThreshold(config, i);
    } else if (config[i] == "expandable_segments") {
      set_expandable_segments_flag = true;
      i = parseExpandableSegments(config, i);
    } else if (config[i] == "base_addr_aligned_kb") {
      i = parseAddrAlignSize(config, i);
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i], PTA_ERROR(ErrCode::PARAM));
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (m_expandable_segments) {
      if (set_expandable_segments_flag) {
          TORCH_CHECK(m_max_split_size == std::numeric_limits<size_t>::max() && m_garbage_collection_threshold == 0,
                      "`max_split_size_mb` or `garbage_collection_threshold`, cannot be enabled with "
                      "`expandable_segments`, please set `expandable_segments` to `False`.",
                      OPS_ERROR(ErrCode::PARAM));
      } else if (m_max_split_size != std::numeric_limits<size_t>::max() || m_garbage_collection_threshold != 0) {
          m_expandable_segments = false;
          TORCH_NPU_WARN_ONCE("`max_split_size_mb` or `garbage_collection_threshold` is enabled, and the "
                              "`expandable_segments` is changed to `False` by default.");
      }
  }
}


class DeviceCachingAllocator {
 private:

  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream
  ska::flat_hash_set<Block*> active_blocks;

  // outstanding acl events
  ska::flat_hash_map<
      c10_npu::NPUStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      npu_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  // record maximum allowed memory.
  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment*> expandable_segments_;

  bool set_fraction = false;

  bool record_history = false;

  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  RecordContext record_context_ = RecordContext::NEVER;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry>*
        alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

 public:

  DeviceCachingAllocator() :
    large_blocks(false),
    small_blocks(true),
    alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = static_cast<int64_t>(CachingAllocatorConfig::max_split_size());
    context_recorder_.store(nullptr);
  }

  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries, RecordContext when)
  {
      std::unique_lock<std::recursive_mutex> lock(mutex);
      TORCH_CHECK(when == RecordContext::NEVER || context_recorder, PTA_ERROR(ErrCode::INTERNAL));
      record_history = enabled;
      context_recorder_.store(record_history ? context_recorder : nullptr);
      alloc_trace_max_entries_ = std::max(size_t(1), alloc_trace_max_entries);
      record_context_ = enabled ? when : RecordContext::NEVER;
      alloc_trace_next = 0;
      alloc_trace->clear();
  }

  bool isHistoryEnabled() { return record_history; }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer)
  {
      oom_observers_.emplace_back(std::move(observer));
  }

  bool checkUceInMemPool()
  {
      auto memUceInfo_ = c10_npu::get_mem_uce_info();
      auto info = memUceInfo_.info;
      const auto all_blocks = get_all_blocks();
      bool any_found = false;
      aclrtMemUceInfo temp_info[memUceInfo_.retSize];
      size_t temp_retsize = 0;

      for (int i = 0; i < memUceInfo_.retSize; ++i) {
          void* addr = info[i].addr;
          size_t length = info[i].len;
          bool found = false;

          // Calculate the start and end address for info[i]
          void* addr_end = static_cast<char*>(addr) + length - 1;

          // Iterate through all blocks and check if there's an overlap with addr
          for (const Block* const head_block : all_blocks) {
              void* block_start = head_block->ptr;
              void* block_end = static_cast<char*>(head_block->ptr) + head_block->size - 1;

              // If there is an overlap, mark the block as unsafe
              if (addr <= block_end && addr_end >= block_start) {
                  const_cast<Block*>(head_block)->is_safe = false;
                  found = true;
                  any_found = true;
                  // Set the unsafe flag only once
                  if (c10_npu::get_npu_data_unsafe_flag() == false) {
                      c10_npu::set_npu_data_unsafe_flag(true);
                  }
              }
          }

          if (found) {
            // update memuceinfo
            temp_info[temp_retsize++] = info[i];
          }
      }

      std::memcpy(memUceInfo_.info, temp_info, temp_retsize * sizeof(aclrtMemUceInfo));
      memUceInfo_.retSize = temp_retsize;

      c10_npu::set_mem_uce_info(memUceInfo_);
      if (!any_found) {
          return false;
      }
      return true;
  }

  void markAllBlockUnsafe()
  {
      for (auto& active_block : active_blocks) {
          active_block->is_safe = false;
      }
      return;
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<c10::GatheredContext> maybeGatherContext(RecordContext level)
  {
      if (record_context_ < level) {
          return nullptr;
      }
      return context_recorder_.load()();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(int device, size_t orig_size, aclrtStream stream, uint8_t allocator_type = 0)
  {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (device == -1) {
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    }

    // process outstanding npuEvents
    process_events(context);
    auto size = round_size(orig_size);
    auto& pool = get_pool(size);

    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found =
      // Search pool
      get_free_block(params) ||
      // Trigger callbacks and retry search
      (trigger_free_memory_callbacks(params) && get_free_block(params));
    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(set_fraction &&
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks(context);
      }
      // Attempt allocate
      block_found = alloc_block(params, false, context, lock) ||
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          (release_available_cached_blocks(params, context) &&
              alloc_block(params, false, context, lock));
    }

    if (!block_found) {
        ASCEND_LOGE(
            "Get a block from the existing pool failed. Try to free cached blocks and reallocate. This error log "
            "can be ignored.");
        // Free all non-split cached blocks and retry alloc.
        c10_npu::NPUWorkspaceAllocator::emptyCache(true);
        block_found = (release_cached_blocks(true, context) && alloc_block(params, true, context, lock));
    }

    if (!block_found) {
      if (params.err == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        size_t device_free;
        size_t device_total;
        NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));

        std::string allowed_info;
        if (set_fraction) {
          allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
        }
        stats.num_ooms += 1;

        record_trace(
            TraceEntry::OOM,
            device_free,
            params.size(),
            params.stream(),
            params.device(),
            std::move(context));
        auto observers_local = oom_observers_;

        // Make sure we do not have the device lock before calling our
        // observers which might need hold the GIL
        // It is safe to release at this point because will no longer
        // be reading any allocator state.

        lock.unlock();

        for (const auto& obs : observers_local) {
            obs(device,
                alloc_size,
                set_fraction ? allowed_memory_maximum : device_total,
                device_free);
        }
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
            format_size(stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
            " current active; ",
            format_size(device_free),
            " free; ",
            allowed_info,
            format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
            " reserved in total by PyTorch)",
            " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.");
      } else {
        NPU_CHECK_ERROR(params.err);
      }
    }

    int64_t ori_block_ptr = int64_t(params.block->ptr);
    size_t align_round = CachingAllocatorConfig::base_addr_aligned_size();
    if (params.size() >= kRoundLarge && CachingAllocatorConfig::expandable_segments() && align_round != 0 &&
        ori_block_ptr % align_round != 0) {
        char* align_ptr = reinterpret_cast<char*>((ori_block_ptr + align_round) - (ori_block_ptr % align_round));
        size_t offset_size = align_ptr - (char*)params.block->ptr;
        if (offset_size + params.size() <= params.block->size) {
            auto size = params.block->size;
            Block* remaining = params.block;

            Block* block = new Block(params.device(), params.stream(), size - offset_size, params.pool, align_ptr);
            block->expandable_segment_ = remaining->expandable_segment_;
            block->next = remaining->next;
            if (block->next) {
                block->next->prev = block;
            }
            block->prev = remaining;

            remaining->next = block;
            remaining->size = offset_size;
            params.pool->blocks.insert(remaining);

            params.block = block;
        }
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        std::move(params), orig_size, std::move(context), split_remainder, allocator_type);
  }

  Block* alloc_found_block(
    AllocParams params,
    size_t orig_size,
    std::shared_ptr<c10::GatheredContext> context,
    bool split_remainder,
    uint8_t allocator_type)
  {
  auto size = params.size();
  auto device = params.device();
  auto pool = params.pool;
  auto stream = params.stream();

  TORCH_INTERNAL_ASSERT(
      params.err == ACL_ERROR_NONE && params.block != nullptr &&
      params.block->ptr != nullptr, PTA_ERROR(ErrCode::PTR));
  Block* block = params.block;
  Block* remaining = nullptr;

  const bool already_split = block->is_split();
  if (split_remainder) {
    remaining = block;

    block = new Block(device, stream, size, pool, block->ptr);
    block->expandable_segment_ = remaining->expandable_segment_;
    block->prev = remaining->prev;
    if (block->prev) {
      block->prev->next = block;
    }
    block->next = remaining;

    remaining->prev = block;
    remaining->ptr = static_cast<char*>(remaining->ptr) + size;
    remaining->size -= size;
    pool->blocks.insert(remaining);

    if (already_split && !block->expandable_segment_) {
      // An already-split inactive block is being shrunk by size bytes.
      update_stat_array(
          stats.inactive_split_bytes,
          -static_cast<std::int64_t>(block->size),
          params.stat_types);
    } else if (!block->expandable_segment_) {
      // A new split inactive block is being created from a previously unsplit
      // block, size remaining->size bytes.
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(
            stats.inactive_split_bytes[stat_type],
            static_cast<std::int64_t>(remaining->size));
        update_stat(stats.inactive_split[stat_type], 1);
      });
    }
  } else if (already_split && !block->expandable_segment_) {
    // An already-split block is becoming active
    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(
          stats.inactive_split_bytes[stat_type],
          -static_cast<std::int64_t>(block->size));
      update_stat(stats.inactive_split[stat_type], -1);
    });
  }

  block->allocated = true;
  block->requested_size = orig_size;
  block->is_safe = true;

  block->context_when_allocated = std::move(context);
  record_trace(
      TraceEntry::ALLOC,
      int64_t(block->ptr),
      orig_size,
      block->stream,
      block->device,
      block->context_when_allocated);

  active_blocks.insert(block);

  for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
    update_stat(stats.allocation[stat_type], 1);
    update_stat(
        stats.allocated_bytes[stat_type],
        static_cast<std::int64_t>(block->size));
    update_stat(stats.active[stat_type], 1);
    update_stat(
        stats.active_bytes[stat_type],
        static_cast<std::int64_t>(block->size));
    update_stat(
        stats.requested_bytes[stat_type],
        static_cast<std::int64_t>(block->requested_size));
  });

  if (block->size >= CachingAllocatorConfig::max_split_size())
    update_stat(stats.oversize_allocations, 1);

  ASCEND_LOGD("PTA CachingAllocator malloc: malloc = %zu, cached = %lu, allocated = %lu",
      block->size,
      stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
      stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);

#ifndef BUILD_LIBTORCH
    torch_npu::profiler::reportMemoryDataToNpuProfiler({
      static_cast<int8_t>(c10::DeviceType::PrivateUse1),
      block->device,
      static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_MALLOC),
      allocator_type,
      reinterpret_cast<int64_t>(block->ptr),
      block->size,
      stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
      stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
      stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
      reinterpret_cast<int64_t>(block->stream)}
    );
#endif

  return block;
}


  void free(Block* block, uint8_t allocator_type = 0)
  {
    std::shared_ptr<c10::GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*(block->pool));
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(stats.allocated_bytes[stat_type], -block->size);
    });

    record_trace(
        TraceEntry::FREE_REQUESTED,
        int64_t(block->ptr),
        block->requested_size,
        block->stream,
        block->device,
        context ? context : block->context_when_allocated);

    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty() && c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
      insert_events(block);
    } else {
      free_block(block, context, allocator_type);
    }

    ASCEND_LOGD("PTA CachingAllocator free: free = %zu, cached = %lu, allocated = %lu",
        orig_block_size,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
#ifndef BUILD_LIBTORCH
    torch_npu::profiler::reportMemoryDataToNpuProfiler({
        static_cast<int8_t>(c10::DeviceType::PrivateUse1),
        block->device,
        static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_FREE),
        allocator_type,
        reinterpret_cast<int64_t>(orig_block_ptr),
        -orig_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        reinterpret_cast<int64_t>(block->stream)}
    );
#endif
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
    block->stream_uses.insert(stream);
  }

  void eraseStream(Block* block, c10_npu::NPUStream stream) {
    std::shared_ptr<c10::GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    block->stream_uses.erase(stream);

    // free block, lazy destory block related events
    for (auto it = npu_events[stream].begin(); it != npu_events[stream].end();) {
      if (block != it->second) {
        it++;
        continue;
      }
      it = npu_events[stream].erase(it);
      block->event_count--;
      if (block->event_count == 0) {
        free_block(block, context);
        break;
      }
    }
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free;
    size_t device_total;
    NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache(bool check_error) {
    std::shared_ptr<c10::GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(check_error, context);
  }

  void release_and_free_events()
  {
      std::unique_lock<std::recursive_mutex> lock(mutex);
      std::shared_ptr<c10::GatheredContext> context = maybeGatherContext(RecordContext::ALL);
      for (auto& st : npu_events) {
          for (auto& e : st.second) {
              EventPool::Event event = std::move(e.first);
              Block* block = e.second;
              block->event_count--;
              if (block->event_count == 0) {
                  free_block(block, context);
              }
          }
      }
      npu_events.clear();
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
      reset_accumulated_stat(stats.requested_bytes[statType]);
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
      reset_peak_stat(stats.requested_bytes[statType]);
    }

    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. **/
  std::vector<SegmentInfo> snapshot()
  {
      std::lock_guard<std::recursive_mutex> lock(mutex);

      size_t total_active = 0;
      std::vector<SegmentInfo> result;
      const auto all_blocks = get_all_blocks();

      for (const Block* const head_block : all_blocks) {
          // For expandable segments, we report one segment for each continguous
          // mapped range of memory
          if (head_block->prev && head_block->prev->mapped) {
              continue;
          }
          result.emplace_back();
          SegmentInfo& segment_info = result.back();
          segment_info.device = head_block->device;
          segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
          segment_info.stream = head_block->stream;
          segment_info.is_large = (!head_block->pool->is_small);
          segment_info.is_expandable = head_block->expandable_segment_;
          segment_info.context_when_allocated =
              head_block->context_when_segment_allocated;

          const Block* block = head_block;
          while (block != nullptr && block->mapped) {
              segment_info.blocks.emplace_back();
              BlockInfo& block_info = segment_info.blocks.back();

              block_info.size = block->size;
              block_info.requested_size = block->requested_size;
              block_info.allocated = block->allocated;
              block_info.active = block->allocated || (block->event_count > 0);

              segment_info.total_size += block_info.size;
              if (block_info.allocated) {
                  segment_info.allocated_size += block_info.size;
              }
              if (block_info.active) {
                  segment_info.active_size += block_info.size;
                  segment_info.requested_size += block_info.requested_size;
              }
              block_info.context_when_allocated = block->context_when_allocated;
              block = block->next;
          }
          total_active += segment_info.active_size;
      }

      std::sort(result.begin(), result.end(),
                [](const SegmentInfo& a, const SegmentInfo& b) {
                    return a.address < b.address;
                });

      record_trace(TraceEntry::SNAPSHOT, 0, total_active, nullptr, 0, nullptr);
      return result;
  }

  std::vector<TraceEntry> trace()
  {
      std::lock_guard<std::recursive_mutex> lock(mutex);
      std::vector<TraceEntry> result;
      result.reserve(alloc_trace->size());
      result.insert(result.end(), alloc_trace->begin() + alloc_trace_next,
                    alloc_trace->end());
      result.insert(result.end(), alloc_trace->begin(),
                    alloc_trace->begin() + alloc_trace_next);

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

 private:

  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  // returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
  Block* find_expandable_block(
      int device,
      aclrtStream stream,
      BlockPool* pool,
      size_t size) {
    Block key(device, stream, 0);

    auto allocatable = [](Block* b) {
      return b && !b->allocated && b->event_count == 0 &&
          b->stream_uses.empty();
    };
    auto has_available_address_space = [&](Block* b) {
      size_t bytes = 0;
      while (bytes < size && allocatable(b)) {
        bytes += b->size;
        b = b->next;
      }
      return bytes >= size;
    };
    for (auto it = pool->unmapped.lower_bound(&key);
         it != pool->unmapped.end() && (*it)->stream == stream;
         ++it) {
      Block* c = *it;
      // we found the lowest address of an unmapped segment
      // but there might be a free segment we can also use
      // right before it
      if (allocatable(c->prev)) {
        c = c->prev;
      }
      if (has_available_address_space(c)) {
        return c;
      }
    }
    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
    expandable_segments_.emplace_back(new ExpandableSegment(
        device, stream, segment_size));

    ExpandableSegment* es = expandable_segments_.back();
    Block* candidate = new Block(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  bool map_block(
      Block* to_map,
      size_t size,
      const std::shared_ptr<c10::GatheredContext>& ctx)
  {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size, PTA_ERROR(ErrCode::VALUE));
    TORCH_INTERNAL_ASSERT(
        !to_map->context_when_allocated); // unmapped blocks should not keep
                                          // history
    auto mapped_range =
        to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    // failed to map the memory
    if (mapped_range.size == 0) {
      return false;
    }
    TORCH_INTERNAL_ASSERT(
        mapped_range.ptr == to_map->ptr && mapped_range.size >= size, PTA_ERROR(ErrCode::INTERNAL));

    BlockPool& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
      // to_map -> remaining -> to_map->next(?)
      Block* remaining = new Block(
          to_map->device,
          to_map->stream,
          to_map->size - mapped_range.size,
          &pool,
          static_cast<char*>(to_map->ptr) + mapped_range.size);
      remaining->mapped = false;
      remaining->expandable_segment_ = to_map->expandable_segment_;
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      to_map->size = mapped_range.size;
    }

    try_merge_blocks(to_map, to_map->prev, pool);
    try_merge_blocks(to_map, to_map->next, pool);

    pool.blocks.insert(to_map);

    // update statistics
    total_allocated_memory += mapped_range.size;
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], mapped_range.size);
    });

    record_trace(
        TraceEntry::SEGMENT_MAP,
        int64_t(mapped_range.ptr),
        mapped_range.size,
        to_map->stream,
        to_map->device,
        ctx);
    if (!to_map->prev && !to_map->context_when_segment_allocated) {
      to_map->context_when_segment_allocated = ctx;
    }

    return true;
  }

  Block* try_allocate_expandable_block(
      int device,
      aclrtStream stream,
      BlockPool* pool,
      size_t size,
      const std::shared_ptr<c10::GatheredContext>& ctx)
  {
    Block* candidate = find_expandable_block(device, stream, pool, size);
    // Candidate is now a list free/unmapped blocks with at least size room:
    // unmapped -> null
    // unmapped -> free -> *
    // free -> unmapped -> *

    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size), ctx)) {
      return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped, PTA_ERROR(ErrCode::INTERNAL));

    while (candidate->size < size) {
      // invariant: free -> unmapped -> *
      // map_block will map some of unmapped and merge with free
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      if (!map_block(new_candidate, std::min(remaining, candidate->next->size), ctx)) {
        return nullptr;
      }
      candidate = new_candidate;
    }
    pool->blocks.erase(candidate);
    return candidate;
  }


  /** moves a block into a pool of cached free blocks **/
  void free_block(
      Block* block,
      const std::shared_ptr<c10::GatheredContext>& context,
      uint8_t allocator_type = 0)
  {
    AT_ASSERT(!block->allocated && block->event_count == 0, PTA_ERROR(ErrCode::VALUE));

    record_trace(
        TraceEntry::FREE_COMPLETED,
        int64_t(block->ptr),
        block->requested_size,
        block->stream,
        block->device,
        context ? context : block->context_when_allocated);

    block->context_when_allocated = nullptr;
    size_t original_block_size = block->size;
    auto orig_block_ptr = block->ptr;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size = static_cast<int64_t>(try_merge_blocks(block, merge_candidate, pool));
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    pool.blocks.insert(block);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segements from
      // inactive_split
      if (!block->expandable_segment_) {
        update_stat(
            stats.inactive_split[stat_type], net_change_inactive_split_blocks);
        update_stat(
            stats.inactive_split_bytes[stat_type],
            net_change_inactive_split_size);
      }
      update_stat(stats.active[stat_type], -1);
      update_stat(stats.active_bytes[stat_type], -original_block_size);
      update_stat(
          stats.requested_bytes[stat_type],
          -static_cast<std::int64_t>(requested_size));
    });
#ifndef BUILD_LIBTORCH
    torch_npu::profiler::reportMemoryDataToNpuProfiler({
        static_cast<int8_t>(c10::DeviceType::PrivateUse1),
        block->device,
        static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_BLOCK_FREE),
        allocator_type,
        reinterpret_cast<int64_t>(orig_block_ptr),
        -original_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        reinterpret_cast<int64_t>(block->stream)}
    );
#endif
  }

  /** combine previously split blocks. returns the size of the subsumed block, or 0 on failure. **/
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split(), PTA_ERROR(ErrCode::VALUE));

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
    auto erased =
        src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
    delete src;
    src = nullptr;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small ||
        CachingAllocatorConfig::expandable_segments()) {
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

    if (C10_UNLIKELY(set_fraction &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.blocks) {
        ++b->gc_count;
      }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      return false;
    }

    if ((*it)->expandable_segment_) {
      if (CachingAllocatorConfig::expandable_segments()) {
        // if we are allocated to the part of the block that is expandable
        // for the purposes of "best fit" we consider its size to be the size it
        // can expand to, not the size it currently is. This means that we
        // sometimes have to search for blocks with bigger 'size' before
        // choosing this segment.
        auto expandable_size = [](Block* b) {
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        auto next = it;
        next++;
        while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
               (*next)->stream == p.stream() &&
               expandable_size(*next) < expandable_size(*it)) {
          it = next++;
        }
      } else {
        // Rarely expandable segments has been turned off after we have
        // already allocated some blocks as expandable. For instance,
        // since we cannot share expandable memory via IPC, someone might
        // temporarily disable it. In this case we need to honor this request
        // by only finding non-expandable blocks
        do {
          it++;
        } while (it != pool.blocks.end() && (*it)->expandable_segment_ &&
                 (*it)->stream == p.stream());
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
          return false;
        }
      }
    }

    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size())) {
          return false;
        }
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) && ((*it)->size >= p.size() + kLargeBuffer)) {
      return false;
    }
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
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

  void garbage_collect_cached_blocks(const std::shared_ptr<c10::GatheredContext>& ctx)
  {
    // Free unused cached blocks to reclaim NPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    c10_npu::npuSynchronizeDevice(true);

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count; // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block, ctx);

          ASCEND_LOGD("PTA CachingAllocator gc: free = %zu, cached = %lu, allocated = %lu",
              block->size,
              stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
              stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
        }
      }
    }
  }

  bool alloc_block(
      AllocParams& p,
      bool isRetry,
      const std::shared_ptr<c10::GatheredContext>& ctx,
      std::unique_lock<std::recursive_mutex>& lock)
  {
    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction && total_allocated_memory + size > allowed_memory_maximum) {
      p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
    } else if (
        CachingAllocatorConfig::expandable_segments()) {
      p.block = try_allocate_expandable_block(
          p.device(), p.stream(), p.pool, p.size(), ctx);
      if (p.block) {
        p.err = ACL_ERROR_NONE;
      } else {
        p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
      }
      return bool(p.block);
    } else {
      p.err = c10_npu::acl::AclrtMallocAlign32(
          &ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    }

    if (p.err != ACL_ERROR_NONE) {
      p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
      return false;
    }
    ASCEND_LOGD("NPUCachingAllocator malloc by AclrtMallocAlign32: size=%zu", size);

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_segments, 1);
    ASCEND_LOGD("pta_memory acl_malloc: malloc = %zu, ret = %d", size, p.err);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    record_trace(
        TraceEntry::SEGMENT_ALLOC,
        int64_t(p.block->ptr),
        p.block->size,
        p.stream(),
        p.device(),
        ctx);
    p.block->context_when_segment_allocated = ctx;
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p,
    const std::shared_ptr<c10::GatheredContext>& ctx)
  {
    if (CachingAllocatorConfig::max_split_size() == std::numeric_limits<size_t>::max()) {
      return false;
    }
    BlockPool &pool = *p.pool;
    Block key = p.search_key;
    key.size =
        (key.size < CachingAllocatorConfig::max_split_size()) ? CachingAllocatorConfig::max_split_size() : key.size;
    auto it = pool.blocks.lower_bound(&key);

    c10_npu::npuSynchronizeDevice(true);

    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks, starting with the largest
      if (it == pool.blocks.begin()) {
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
          release_block(*cur, ctx);
        } else {
          release_block(*cur, ctx);
          break;
        }
      }
      if (totalReleased < key.size) {
        return false;
      }
    } else {
      release_block(*it, ctx);
    }
    return true;
  }

  bool release_cached_blocks(bool check_error, const std::shared_ptr<c10::GatheredContext>& context)
  {
      // Make sure event deque from taskqueue, then synchronize Event
      c10_npu::npuSynchronizeDevice(check_error);

      // First ensure that all blocks that can't currently be allocated due to
      // outstanding events are returned to the pool.
      synchronize_and_free_events(check_error, context);

      // Free all non-split cached blocks
      release_blocks(large_blocks, context);
      release_blocks(small_blocks, context);

      return true;
  }

  void release_expandable_segment(Block* block) {
    TORCH_INTERNAL_ASSERT(
        block->size == block->expandable_segment_->size(),
        "block disagrees with segment", PTA_ERROR(ErrCode::INTERNAL));
    TORCH_INTERNAL_ASSERT(!block->mapped, PTA_ERROR(ErrCode::INTERNAL));
    auto it = std::find(
        expandable_segments_.begin(),
        expandable_segments_.end(),
        block->expandable_segment_);
    TORCH_INTERNAL_ASSERT(it != expandable_segments_.end(), PTA_ERROR(ErrCode::INTERNAL));
    expandable_segments_.erase(it);
    block->pool->unmapped.erase(block);
    delete block->expandable_segment_;
    block->expandable_segment_ = nullptr;
    delete block;
    block = nullptr;
  }

  void release_block(
      Block* block,
      const std::shared_ptr<c10::GatheredContext>& context)
  {
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_, PTA_ERROR(ErrCode::VALUE));
    ASCEND_LOGD("NPUCachingAllocator free by aclrtFree: size=%zu", block->size);

    record_trace(
        TraceEntry::SEGMENT_FREE,
        int64_t(block->ptr),
        block->size,
        block->stream,
        block->device,
        context ? context : block->context_when_segment_allocated);

    aclrtFree((void*)block->ptr);
    total_allocated_memory -= block->size;

    auto* pool = block->pool;

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(stats.reserved_bytes[stat_type], -block->size);
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);

    ASCEND_LOGD("pta_memory acl_free: free_size = %zu", block->size);

    pool->blocks.erase(block);
    delete block;
    block = nullptr;
    }

  void unmap_block(
      Block* block,
      const std::shared_ptr<c10::GatheredContext>& context)
  {
    auto unmapped = block->expandable_segment_->unmap(
        SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
      return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size =
        static_cast<char*>(unmapped.ptr) - static_cast<char*>(block->ptr);
    if (before_size > 0) {
      // prev? -> before_free -> block
      Block* before_free = new Block(
          block->device, block->stream, before_size, block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->blocks.insert(before_free);
    }

    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // block -> after_free -> next?
      Block* after_free = new Block(
          block->device,
          block->stream,
          after_size,
          block->pool,
          static_cast<char*>(unmapped.ptr) + unmapped.size);
      after_free->expandable_segment_ = block->expandable_segment_;
      after_free->splice(block, block->next);
      block->pool->blocks.insert(after_free);
    }

    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    try_merge_blocks(block, block->prev, *block->pool);
    try_merge_blocks(block, block->next, *block->pool);
    block->pool->unmapped.insert(block);

    // update statistics
    total_allocated_memory -= unmapped.size;
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], -unmapped.size);
    });

    record_trace(
        TraceEntry::SEGMENT_UNMAP,
        int64_t(unmapped.ptr),
        unmapped.size,
        block->stream,
        block->device,
        context ? context : block->context_when_segment_allocated);
  }

  void release_blocks(
      BlockPool& pool,
      const std::shared_ptr<c10::GatheredContext>& context)
  {
    std::vector<Block*> to_unmap;
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block *block = *it;
      ++it;
      if (block->expandable_segment_) {
        // unmapping will mutate the free pool
        // so just gather what needs to be freed
        // to avoid invalidating the iterator
        to_unmap.push_back(block);
      } else if (!block->prev && !block->next) {
        release_block(block, context);
      }
    }
    for (Block* block : to_unmap) {
      unmap_block(block, context);
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
  }

  EventPool::Event create_event_internal(int idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  void synchronize_and_free_events(bool check_error, const std::shared_ptr<c10::GatheredContext>& context)
  {
    // Synchronize on outstanding events and then free associated blocks.
    for (auto& st : npu_events) {
      for (auto& e : st.second) {
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;
        if (check_error) {
          NPU_CHECK_ERROR(aclrtSynchronizeEvent(*event));
        } else {
          NPU_CHECK_WARN(aclrtSynchronizeEvent(*event));
        }
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event.get()));
        }
#endif
        ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", event.get());

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
    }

    npu_events.clear();
  }

  void insert_events(Block* block) {
    aclrtContext compiler_ctx = aclrtContext();
    aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);
    NPU_CHECK_ERROR(aclrtSetCurrentContext(c10_npu::GetDeviceContext(block->device)));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty(), PTA_ERROR(ErrCode::VALUE));
    for (auto& stream : streams) {
      NPU_CHECK_ERROR(c10_npu::SetDevice(stream.device_index()));

      EventPool::Event event = create_event_internal(stream.device_index());
      event->record(stream);
      ASCEND_LOGI("Event: record DeviceAllocator is successfully executed, event=%p", event.get());

      block->event_count++;
      npu_events[stream].emplace_back(std::move(event), block);
    }
    if (ret_ctx == ACL_ERROR_NONE) {
      NPU_CHECK_ERROR(aclrtSetCurrentContext(compiler_ctx));
    }
  }

  void process_events(const std::shared_ptr<c10::GatheredContext>& context)
  {
    // Process outstanding npuEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    for (auto it = npu_events.begin(); it != npu_events.end();) {
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        if (!event->query()) {
          e.first = std::move(event);
          break;
        }

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = npu_events.erase(it);
      } else {
        it++;
      }
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

  void record_trace(
      TraceEntry::Action action,
      int64_t addr,
      size_t size,
      aclrtStream stream,
      int device,
      std::shared_ptr<c10::GatheredContext> context)
  {
    if (!record_history) {return;}

    auto te = TraceEntry(
        action,
        device,
        addr,
        size,
        stream,
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);

    if (record_history) {
      if (alloc_trace->size() < alloc_trace_max_entries_) {
        alloc_trace->emplace_back(te);
      } else {
        (*alloc_trace)[alloc_trace_next++] = te;
        if (alloc_trace_next == alloc_trace_max_entries_) {
          alloc_trace_next = 0;
        }
      }
    }
  }

};

static void uncached_delete(void* ptr)
{
    if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        c10_npu::npuSynchronizeDevice(false);
    }
    ASCEND_LOGD("Without NPUCachingAllocator, free by aclrtFree.");
    NPU_CHECK_ERROR(aclrtFree(ptr));
}

void local_raw_delete(void* ptr);

class NpuCachingAllocator : public NPUAllocator {
 private:

  std::mutex mutex;

  // allocated blocks by device pointer
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:

  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  Block* get_allocated_block(void* ptr, bool remove = false) {
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

  void init(int device_count) override
    {
    int size = static_cast<int>(device_allocator.size());
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  bool initialized() override
    {
        return !device_allocator.empty();
    }
  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, aclrtStream stream) {
      TORCH_INTERNAL_ASSERT(0 <= device && static_cast<size_t>(device) < device_allocator.size(),
                            "Allocator not initialized for device ", device, ": did you call init?",
                            PTA_ERROR(ErrCode::PARAM));
      Block* block = device_allocator[device]->malloc(device, size, stream);

    add_allocated_block(block);
    *devPtr = static_cast<void*>(block->ptr);
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuMemoryAllocation(
            reinterpret_cast<uintptr_t>(*devPtr));
    }
#endif
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
    if (C10_UNLIKELY(trigger)) {
        trigger->traceNpuMemoryDeallocation(
            reinterpret_cast<uintptr_t>(block->ptr));
    }
#endif
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;
    device_allocator[block->device]->free(block);
  }

  void setMemoryFraction(double fraction, int device) override
  {
    TORCH_INTERNAL_ASSERT(
        0 <= device && device < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?", PTA_ERROR(ErrCode::PARAM));
    TORCH_INTERNAL_ASSERT(
        0 <= fraction  && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).", PTA_ERROR(ErrCode::PARAM));

    c10_npu::SetDevice(device);

    device_allocator[device]->setMemoryFraction(fraction);
  }

  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries,
                     RecordContext when) override
  {
      for (auto& allocator : device_allocator) {
          allocator->recordHistory(enabled, context_recorder,
                                   alloc_trace_max_entries, when);
      }
  }

  bool isHistoryEnabled() override
  {
      int device = 0;
      NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
      return device_allocator[device]->isHistoryEnabled();
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override
  {
      for (auto& allocator : device_allocator) {
          allocator->attachOutOfMemoryObserver(std::move(observer));
      }
  }

  bool checkUceInMemPool(int device) override
  {
      return device_allocator[device]-> checkUceInMemPool();
  }

  bool checkBlockIsSafe(const c10::DataPtr& ptr) override
  {
      if (!ptr.get()) {
          return true;
      }
      if (ptr.get_deleter() != &local_raw_delete) {
          return true;
      }
      Block* block = get_allocated_block(ptr.get());
      TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found", PTA_ERROR(ErrCode::NOT_FOUND));
      return block->is_safe;
  }

  void markAllBlockUnsafe(int device) override
  {
      return device_allocator[device]-> markAllBlockUnsafe();
  }

  void updateBlockToSafe(const c10::DataPtr &ptr) override
  {
      if (!ptr.get()) {
          return;
      }
      if (ptr.get_deleter() != &local_raw_delete) {
          return;
      }
      Block* block = get_allocated_block(ptr.get());
      TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found", PTA_ERROR(ErrCode::NOT_FOUND));
      block->is_safe = true;
  }

  void cleanEvent() override
  {
      int count = static_cast<int>(device_allocator.size());
      for (int i = 0; i < count; i++)
          device_allocator[i]->release_and_free_events();
  }

  void emptyCache(bool check_error) override
  {
    int count = static_cast<int>(device_allocator.size());
    for (int i = 0; i < count; i++)
      device_allocator[i]->emptyCache(check_error);
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) override
  {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) override
  {
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
    if (ptr.get_deleter() != &local_raw_delete) {
      return;
    }

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found", PTA_ERROR(ErrCode::NOT_FOUND));
    device_allocator[block->device]->recordStream(block, stream);
  }

  void eraseStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream)
  {
      if (!ptr.get()) {
          return;
      }

      // If a tensor is not allocated by this instance, simply skip
      // This usually happens when NPU tensors are shared across processes,
      // we have implemented reference counting based sharing mechanism to
      // guarantee tensors won't be accidentally freed by one process while
      // they are still being used in another
      if (ptr.get_deleter() != &local_raw_delete) {
          TORCH_NPU_WARN_ONCE("Tensor not is not allocated by NPUCachingAllocator, skip eraseStream.");
          return;
      }

      Block* block = get_allocated_block(ptr.get());
      if (!block) {
          AT_ERROR("invalid device pointer: ", ptr.get());
      }

      if (block->stream != c10_npu::getCurrentNPUStream(block->device).stream(false)) {
          // If the Stream applying for tensor block different from
          // the stream of submiting event wait task in HCCL synchronize()
          // method, the recordSteam can not be erased.
          // New tensor creation may use the block before HCCL op is complete.
          return;
      }

      device_allocator[block->device]->eraseStream(block, stream);
  }

  SnapshotInfo snapshot() override
  {
    SnapshotInfo result;
    int count = static_cast<int>(device_allocator.size());
    for (int i = 0; i < count; i++) {
      result.device_traces.emplace_back(device_allocator[i]->trace());
      auto snap = device_allocator[i]->snapshot();
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }
    return result;
  }

  c10::DataPtr allocate(size_t size) override
  {
      int device = 0;
      NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
      void* devPtr = nullptr;
      void (*deleteFunc)(void*) = &local_raw_delete;

      if (size != 0) {
          if (c10_npu::option::OptionsManager::CheckForceUncached()) {
              deleteFunc = &uncached_delete;
              size_t alloc_size = size + 32;
              NPU_CHECK_ERROR(c10_npu::acl::AclrtMallocAlign32(&devPtr, alloc_size,
                                                               aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
              ASCEND_LOGD("Without NPUCachingAllocator, malloc by "
                          "AclrtMallocAlign32: size=%zu",
                          alloc_size);
          } else {
              this->malloc(&devPtr, device, size, c10_npu::getCurrentNPUStreamNoWait(device));
          }
      }
      return {devPtr, devPtr, deleteFunc, c10::Device(c10::DeviceType::PrivateUse1, device)};
  }

  c10::DeleterFnPtr raw_deleter() const override
  {
      if (c10_npu::option::OptionsManager::CheckForceUncached()) {
          return &uncached_delete;
      } else {
          return &local_raw_delete;
      }
  }

  void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) override
  {
      device_allocator[dev_id]->cacheInfo(cachedAndFree, largestBlock);
  }

  void assertValidDevice(int device)
  {
      const auto device_num = device_allocator.size();
      TORCH_CHECK(0 <= device && device < static_cast<int64_t>(device_num), "Invalid device argument ", device,
                  ": did you call init?", PTA_ERROR(ErrCode::PARAM));
  }

  DeviceStats getDeviceStats(int device) override
  {
      assertValidDevice(device);
      return device_allocator[device]->getStats();
  }

  void resetAccumulatedStats(int device) override
  {
      assertValidDevice(device);
      device_allocator[device]->resetAccumulatedStats();
  }

  void resetPeakStats(int device) override
  {
      assertValidDevice(device);
      device_allocator[device]->resetPeakStats();
  }

  void* raw_alloc(size_t nbytes) override
  {
    if (nbytes == 0) {
      return nullptr;
    }
    int device = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, c10_npu::getCurrentNPUStreamNoWait(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream) override
  {
    if (nbytes == 0) {
      return nullptr;
    }
    int device;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, stream);
    return r;
  }

  void raw_delete(void* ptr) override
  {
    this->free(ptr);
  }

  void FreeDeviceCachedMemory(int device) override
  {
    device_allocator[device]->emptyCache(true);
  }

  std::string name() override
  {
    return "native";
  }

  // Note [COW/lazy_clone is not supported yet]
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

NpuCachingAllocator caching_allocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &caching_allocator);


void local_raw_delete(void* ptr)
{
  caching_allocator.free(ptr);
}

void* MallocBlock(size_t size, void *stream, int device) {
  if (device == -1) {
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
  }
  if ((device < 0) || (device > static_cast<int>(caching_allocator.device_allocator.size()))) {
    return nullptr;
  }
  AT_ASSERT(caching_allocator.device_allocator[device], PTA_ERROR(ErrCode::NOT_FOUND));
  AT_ASSERT(stream, PTA_ERROR(ErrCode::NOT_FOUND));
  auto block = caching_allocator.device_allocator[device]->malloc(device, size, stream,
    static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_EXTERNAL));
  AT_ASSERT(block, PTA_ERROR(ErrCode::NOT_FOUND));
  return reinterpret_cast<void*>(block);
}

void FreeBlock(void *handle) {
  Block* block = reinterpret_cast<Block*>(handle);
  AT_ASSERT(block, PTA_ERROR(ErrCode::PTR));
  caching_allocator.assertValidDevice(block->device);
  AT_ASSERT(caching_allocator.device_allocator[block->device], PTA_ERROR(ErrCode::NOT_FOUND));
  caching_allocator.device_allocator[block->device]->free(block,
    static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_EXTERNAL));
}

void* GetBlockPtr(const void *handle) {
  const Block* block = reinterpret_cast<const Block*>(handle);
  AT_ASSERT(block, PTA_ERROR(ErrCode::PTR));
  return block->ptr;
}

size_t GetBlockSize(const void *handle) {
  const Block* block = reinterpret_cast<const Block*>(handle);
  AT_ASSERT(block, PTA_ERROR(ErrCode::PTR));
  return block->size;
}

struct BackendStaticInitializer {
    BackendStaticInitializer()
    {
      allocator.store(&caching_allocator);
    }
};

std::atomic<NPUAllocator*> allocator;
BackendStaticInitializer backend_static_initializer;

std::mutex* getFreeMutex() {
  static std::mutex npu_free_mutex;
  return &npu_free_mutex;
}

} // namespace NPUCachingAllocator
} // namespace c10_npu
