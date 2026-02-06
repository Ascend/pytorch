#include <algorithm>
#include <bitset>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <vector>
#include <fstream>

#include <c10/core/Allocator.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/llvmMathExtras.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/core/npu/NPURecovery.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "NPUBlockHandle.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

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

constexpr size_t kMinBlockSize = 512;                 // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;                // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;              // "small" allocations are packed in 2 MiB blocks
constexpr size_t kExtraLargeBuffer = 1073741824;      // "extra large" allocations may be packed in 1 GB blocks
constexpr size_t kMinLargeAlloc = 10485760;           // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;               // round up large allocs to 2 MiB
constexpr size_t kSmallPoolVirAddrSize = 2147483648;  // 2 GB
constexpr size_t kLargePoolVirAddrSize = 10737418240; // 10 GB
const std::string kMinCannVersion = "8.1.RC1";        // minimum cann version which supports 1g mem 8.1.RC1
const std::string kMinDriverVersion = "25.0.RC1";     // minimum driver version which supports 1g mem 25.0.RC1
const std::string kCannModule = "CANN";               // cann module name
constexpr int kPrecision = 4;                         // precision of the memory usage information

static char SHAREABLE_HANDLE_VERSION = 1;
enum ShareableHandleType : char {
    SHAREABLE_NPU_MALLOC = 'c',
    SHAREABLE_NPU_EXPANDABLE_SEGMENT = 'e'
};

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat &stat, int64_t amount)
{
    stat.current += amount;
    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0) {
        stat.allocated += amount;
    }
    if (amount < 0) {
        stat.freed += -amount;
    }
}

void reset_accumulated_stat(Stat &stat)
{
    stat.allocated = 0;
    stat.freed = 0;
}

void reset_peak_stat(Stat &stat)
{
    stat.peak = stat.current;
}

template <typename Func> void for_each_selected_stat_type(const StatTypes &stat_types, Func f)
{
    for (const auto stat_type : c10::irange(stat_types.size())) {
        if (stat_types[stat_type]) {
            f(stat_type);
        }
    }
}

void update_stat_array(StatArray &stat_array, int64_t amount, const StatTypes &stat_types)
{
    for_each_selected_stat_type(stat_types,
        [&stat_array, amount](size_t stat_type) { update_stat(stat_array[stat_type], amount); });
}

bool IsMallocPage1GMem(bool is_small_pool)
{
    static bool is_support_page_size_1g = []() {
        if (!c10_npu::NPUCachingAllocator::isConfig1GPageSizeEnable()) {
            return false;
        }

        if (!IsGteCANNVersion(kMinCannVersion, kCannModule)) {
            TORCH_NPU_WARN_ONCE("The application for 1G large-page physical memory failed. "
                "Using the HUGE_MEM memory page allocation method may result in performance degradation. "
                "This warning occurs because the PYTORCH_NPU_ALLOC_CONF = page_size:1g configuration is enabled, "
                "but the current driver version does not support this feature. "
                "Please upgrade the CANN package version.");
            return false;
        }

        if (!IsGteDriverVersion(kMinDriverVersion)) {
            TORCH_NPU_WARN_ONCE("The application for 1G large-page physical memory failed. "
                "Using the HUGE_MEM memory page allocation method may result in performance degradation. "
                "This warning occurs because the PYTORCH_NPU_ALLOC_CONF = page_size:1g configuration is enabled, "
                "but the current driver version does not support this feature. "
                "Please upgrade the HDK(driver) package version.");
            return false;
        }
        return true;
    }();

    return !is_small_pool && is_support_page_size_1g;
}

size_t AddPadSize()
{
    static size_t add_size = -1;
    if (add_size == -1) {
        if (GetSocVersion() >= SocVersion::Ascend910_95) {
            add_size = 0;
        } else {
            add_size = 32;
        }
    }
    return add_size;
}

struct Block;
struct PrivatePool;
using Comparison = bool (*)(const Block *, const Block *);
static bool BlockComparatorSize(const Block *a, const Block *b);
static bool BlockComparatorAddress(const Block *a, const Block *b);

struct BlockPool {
    std::set<Block *, Comparison> blocks;
    std::set<Block *, Comparison> unmapped;
    const bool is_small;
    PrivatePool *owner_PrivatePool;
    // store unmapped handles
    std::vector<aclrtDrvMemHandle> free_physical_handles_;

    BlockPool(bool small, PrivatePool *private_pool = nullptr)
        : blocks(BlockComparatorSize),
          unmapped(BlockComparatorAddress),
          is_small(small),
          owner_PrivatePool(private_pool)
    {}
};

struct ExpandableSegment;

struct Block {
    int device;             // npu
    aclrtStream stream;     // allocation stream
    stream_set stream_uses; // streams on which the block was used
    size_t size;            // block size in bytes
    size_t requested_size;  // memory originally requested
    BlockPool *pool;        // owning memory pool
    void *ptr;              // memory address
    bool allocated;         // in-use flag
    bool mapped{ true };    // is the virtual address range this Block references
                            // backed by physical pages. Always true when
                            // expandable_segment_ is null. When false
                            // This Block will be aligned to the segment size
                            // of its expandable_segment_.
    Block *prev;            // prev block if split from a larger allocation
    Block *next;            // next block if split from a larger allocation
    int event_count;        // number of outstanding NPU events
    int gc_count{ 0 };      // counter for prioritizing older / less useful blocks for
                            // garbage collection
    ExpandableSegment *expandable_segment_{ nullptr };
    bool is_safe{ true };
    void* hccl_work_ptr{ nullptr };
    std::shared_ptr<c10::GatheredContext> context_when_allocated;
    // only set for the first block in the segment (when prev == null)
    // this records the frame information when aclMalloc was called
    // whereas context_when_allocated records the last time we handed this
    // memory out from our cache.
    std::shared_ptr<c10::GatheredContext> context_when_segment_allocated;

    Block(int device, aclrtStream stream, size_t size, BlockPool *pool, void *ptr)
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
          gc_count(0)
    {}

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
          gc_count(0)
    {}

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }

    void splice(Block *before, Block *after)
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
    char *ptr;
    size_t size;
    SegmentRange(void *p, size_t s) : ptr(static_cast<char *>(p)), size(s) {}
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
        std::optional<aclrtStream> stream,
        size_t size,
        size_t handleNum = 0)
        : device_(device),
          stream_(stream),
          max_handles_(handleNum * 2),
          // 2MB for small pool, 20MB for large pool
          segment_size_(size)
    {
        if (max_handles_ == 0) {
            size_t device_free;
            size_t device_total;
            NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
            // we allocate enough address space for 1 1/8 the total memory on the NPU.
            // This allows for some cases where we have to unmap pages earlier in the
            // segment to put them at the end.
            max_handles_ = numSegments(device_total + device_total / 8);
            if (c10_npu::option::OptionsManager::IsHcclZeroCopyEnable()) {
                // prevent HCCL reserve virtual address out of memory
                // small pool reserve 2G
                // non-default stream large pool 10G
                auto default_stream = c10_npu::getDefaultNPUStream().stream(false);
                if (kSmallBuffer == segment_size_) {
                    max_handles_ = numSegments(kSmallPoolVirAddrSize);
                } else if (default_stream != *stream) {
                    max_handles_ = numSegments(kLargePoolVirAddrSize);
                }
            }
        }

        NPU_CHECK_ERROR(
            c10_npu::acl::AclrtReserveMemAddress(&ptr_, segment_size_ * max_handles_, 0, nullptr, 1, getHcclComm()));
        ASCEND_LOGD("NPUCachingAllocator malloc by AclrtReserveMemAddress: size=%zu, segment_size=%zu",
            segment_size_ * max_handles_, segment_size_);
    }
    // begin must be aligned to segment_size_.
    // returns the actual range mapped, which may be
    // greater than requested if size is not aligned to segment_size_.
    // return size of 0 indicates OOM
    SegmentRange map(SegmentRange range, BlockPool *pool)
    {
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
            if (!pool->free_physical_handles_.empty()) {
                ASCEND_LOGD("Remap cached physical handles for block %zu", i);
                handle = pool->free_physical_handles_.back();
                pool->free_physical_handles_.pop_back();
                handles_.at(i) = Handle{handle, std::nullopt};
                continue;
            }
            aclrtPhysicalMemProp prop = {};
            prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr = (segment_size_ == kExtraLargeBuffer) ? ACL_HBM_MEM_HUGE1G : ACL_HBM_MEM_HUGE;
            prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = static_cast<unsigned>(device_);
            prop.reserve = 0;
            ASCEND_LOGD("Alloc memory from physical device for block %zu", i);
            auto status = c10_npu::acl::AclrtMallocPhysical(&handle, segment_size_, &prop, 0);
            if (status == ACL_ERROR_RT_MEMORY_ALLOCATION) {
                for (auto j : c10::irange(begin, i)) {
                    auto h = handles_.at(j).value();
                    handles_.at(j) = c10::nullopt;
                    NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h.handle));
                }
                trimHandles();
                return rangeFromHandles(begin, begin);
            }
            NPU_CHECK_ERROR(status, "aclrtMallocPhysical");
            handles_.at(i) = Handle{handle, std::nullopt};
        }
        for (auto i : c10::irange(begin, end)) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtMapMem((char *)ptr_ + i * segment_size_, segment_size_, 0,
                handles_.at(i).value().handle, 0, getHcclComm()));
        }
        ASCEND_LOGD("NPUCachingAllocator map: segment_size=%zu", segment_size_);
        return rangeFromHandles(begin, end);
    }

    // unmaps all the completely empty segment_size_ segments between
    // [begin, begin + size), returns the offset where the range begin,
    // and the actual size unmapped (multiple of segment_size_)
    SegmentRange unmap(SegmentRange range, BlockPool *pool)
    {
        auto begin = segmentRight(range.ptr);
        auto end = segmentLeft(range.ptr + range.size);
        if (begin >= end) {
            return SegmentRange{ range.ptr, 0 };
        }
        unmapHandles(begin, end, pool);
        return rangeFromHandles(begin, end);
    }

    // Setup IPC sharing for range.
    // Returns the (larger) range that was actually shared.
    // Serializes data to std::ostream that can be passed to the
    // other process, and then restored as an exapandable segment
    // via ExpandableSegment::fromShared(istream);
    SegmentRange share(SegmentRange range, std::ostream& buf)
    {
        auto begin = segmentLeft(range.ptr);
        auto end = segmentRight(range.ptr + range.size);
        ShareHeader header{segment_size_, end - begin};
        buf.write((const char*)&header, sizeof(ShareHeader));
        for (auto i : c10::irange(begin, end)) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            auto& handle = handles_.at(i).value();
            if (!handle.shareableHandle) {
                uint64_t shareableHandle = 0;
                NPU_CHECK_ERROR(c10_npu::acl::AclrtMemExportToShareableHandle(
                    handle.handle, ACL_MEM_HANDLE_TYPE_NONE, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION, &shareableHandle));
                handle.shareableHandle = shareableHandle;
            }
            uint64_t shandle = *handle.shareableHandle;
            buf.write((const char*)&shandle, sizeof(uint64_t));
        }
        return rangeFromHandles(begin, end);
    }

    static std::unique_ptr<ExpandableSegment> fromShared(
        c10::DeviceIndex device,
        std::istream& buf)
    {
        ShareHeader header{};
        buf.read((char*)&header, sizeof(ShareHeader));
        auto segment = std::make_unique<ExpandableSegment>(
            device,
            std::nullopt,
            header.segment_size,
            header.num_handles);
        for (auto i : c10::irange(header.num_handles)) {
            (void)i;
            uint64_t shareableHandle = 0;
            buf.read((char*)&shareableHandle, sizeof(uint64_t));

            int32_t deviceId = static_cast<int32_t>(device);
            aclrtDrvMemHandle handle;
            NPU_CHECK_ERROR(c10_npu::acl::AclrtMemImportFromShareableHandle(
                shareableHandle, deviceId, &handle));
            segment->handles_.emplace_back(Handle{handle, shareableHandle});
        }
        segment->mapAndSetAccess(0, header.num_handles);
        return segment;
    }

    char *ptr() const
    {
        return (char *)ptr_;
    }

    size_t size() const
    {
        return max_handles_ * segment_size_;
    }

    void setHcclComm(std::shared_ptr<c10d_npu::HCCLComm> hcclComm)
    {
        TORCH_INTERNAL_ASSERT(hcclComm, "hcclComm is null.", PTA_ERROR(ErrCode::INTERNAL));
        hcclComm_ = hcclComm;
        HCCL_CHECK_ERROR(at_npu::hccl::HcclCommSetMemoryRangeFace(hcclComm_->getHcclComm(), ptr_,
            segment_size_ * max_handles_, 0, 1));
        for (auto i : c10::irange(handles_.size())) {
            HCCL_CHECK_ERROR(at_npu::hccl::HcclCommActivateCommMemoryFace(hcclComm_->getHcclComm(),
                (char *)ptr_ + i * segment_size_, segment_size_, 0, handles_.at(i).value().handle, 0));
        }
    }

    ~ExpandableSegment()
    {
        forEachAllocatedRange([&](size_t begin, size_t end) { unmapHandles(begin, end); });
        NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr_, getHcclComm()));
        ASCEND_LOGD("NPUCachingAllocator free by AclrtReleaseMemAddress");
    }

private:
    void mapAndSetAccess(size_t begin, size_t end)
    {
        for (auto i : c10::irange(begin, end)) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtMapMem((char *)ptr_ + i * segment_size_, segment_size_, 0,
                handles_.at(i).value().handle, 0, getHcclComm()));
        }
        ASCEND_LOGD("NPUCachingAllocator mapAndSetAccess: segment_size=%zu", segment_size_);
    }

    void unmapHandles(size_t begin, size_t end, BlockPool *pool = nullptr)
    {
        // note: unlike aclrtFree, MemUnmap and MemRelease do
        // not appear to synchronize in all cases, so we have to wait for the
        // stream to finish before this memory is truly free.

        // cannot call c10::npu::stream_synchronize because
        // it might grab the GIL which can lead to a deadlock
        // Locking order must be GIL -> Allocator Lock
        {
            c10_npu::NPUGuard device_guard(device_);
            c10_npu::npuSynchronizeDevice(true);
        }
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuStreamSynchronization(reinterpret_cast<uintptr_t>(*stream_));
        }
#endif
        for (auto i : c10::irange(begin, end)) {
            Handle h = handles_.at(i).value();
            handles_.at(i) = c10::nullopt;
            NPU_CHECK_ERROR(c10_npu::acl::AclrtUnmapMem((char *)ptr_ + segment_size_ * i, getHcclComm()));
            if (!pool) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(h.handle));
            } else {
                pool->free_physical_handles_.push_back(h.handle);
            }
        }
        ASCEND_LOGD("NPUCachingAllocator unmap: segment_size=%zu", segment_size_);
        trimHandles();
    }

    void trimHandles()
    {
        while (!handles_.empty() && !handles_.back()) {
            handles_.pop_back();
        }
    }

    void forEachAllocatedRange(std::function<void(size_t, size_t)> fn)
    {
        size_t start = 0;
        for (auto i : c10::irange(handles_.size())) {
            if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
                start = i;
            }
            if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
                fn(start, i + 1);
            }
        }
    }

    size_t numSegments(size_t size)
    {
        return (size + segment_size_ - 1) / segment_size_;
    }

    size_t segmentLeft(char *p)
    {
        auto size = p - ptr();
        return static_cast<size_t>(size) / segment_size_;
    }

    size_t segmentRight(char *p)
    {
        auto size = p - ptr();
        return numSegments(size);
    }

    SegmentRange rangeFromHandles(size_t begin, size_t end)
    {
        return SegmentRange(ptr() + segment_size_ * begin, segment_size_ * (end - begin));
    }

    HcclComm getHcclComm()
    {
        if (hcclComm_) {
            return hcclComm_->getHcclComm();
        }
        return nullptr;
    }

    int device_;
    std::optional<aclrtStream> stream_;
    void *ptr_{};
    size_t max_handles_;
    size_t segment_size_;
    struct Handle {
        aclrtDrvMemHandle handle;
        std::optional<uint64_t> shareableHandle;
    };
    struct ShareHeader {
        size_t segment_size;
        size_t num_handles;
    };
    std::vector<std::optional<Handle>> handles_;
    std::shared_ptr<c10d_npu::HCCLComm> hcclComm_;
};

// BlockState, BlockPoolState, and PrivatePoolState contain the information
// needed to reconstruct a private pool to a previous state. See note
// [Checkpointing PrivatePoolState]
struct BlockState {
  c10::DeviceIndex device = 0;
  aclrtStream stream = nullptr;
  stream_set stream_uses = {};
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int64_t gc_count_base = 0;
  // maintain invariant that event_count == 0 ;
  // history will be left alone in checkpoint

  explicit BlockState(Block* block);
};

struct SegmentState {
  std::vector<BlockState> blocks;
  bool is_small = false;

  explicit SegmentState(Block* head);
};

struct PrivatePoolState : AllocatorState {
  // omitting use_count, and aclMalloc_count as they remain the same
  MempoolId_t owner_id = {0, 0};

  std::vector<SegmentState> segments;

  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};

struct RestoreResult {
  std::vector<void*> allocations_freed;
  std::vector<Block*> allocations_created;
};

static bool BlockComparatorSize(const Block *a, const Block *b)
{
    if (a->stream != b->stream) {
        return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

static bool BlockComparatorAddress(const Block *a, const Block *b)
{
    if (a->stream != b->stream) {
        return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
    }
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

struct AllocParams {
    AllocParams(int device, size_t size, aclrtStream stream, BlockPool *pool, size_t alloc_size, DeviceStats &stats)
        : search_key(device, stream, size), pool(pool), alloc_size(alloc_size), block(nullptr), err(ACL_ERROR_NONE)
    {}

    int device() const
    {
        return search_key.device;
    }
    aclrtStream stream() const
    {
        return search_key.stream;
    }
    size_t size() const
    {
        return search_key.size;
    }

    Block search_key;
    BlockPool *pool;
    size_t alloc_size;
    Block *block;
    StatTypes stat_types = { false };
    aclError err;
};

class EventPool {
public:
    using Event = std::unique_ptr<c10_npu::NPUEvent, std::function<void(c10_npu::NPUEvent *)>>;
    // Explicit device count
    EventPool() : pools_(c10_npu::device_count()) {}

    Event get(int device)
    {
        TORCH_INTERNAL_ASSERT(0 <= device, PTA_ERROR(ErrCode::VALUE));
        TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()), PTA_ERROR(ErrCode::VALUE));
        auto &pool = pools_[device];
        auto destructor = [&pool](c10_npu::NPUEvent *event) {
            std::lock_guard<std::mutex> g(pool.mutex_);
            pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
        };

        // Try to acquire an event from the per-device pool.
        {
            std::lock_guard<std::mutex> g(pool.mutex_);
            if (!pool.event_pool_.empty()) {
                auto *event = pool.event_pool_.back().release();
                pool.event_pool_.pop_back();
                return Event(event, destructor);
            }
        }
        // otherwise, allocate a new event that will be returned to the pool on
        // destruction.
        return Event(std::make_unique<c10_npu::NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS).release(), destructor);
    }

    void empty_cache()
    {
        for (auto &pool : pools_) {
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

// NPU graphs helper
struct PrivatePool {
    PrivatePool() : large_blocks(false, this), small_blocks(true, this) {}
    PrivatePool(const PrivatePool &) = delete;
    PrivatePool(PrivatePool &&) = delete;
    PrivatePool &operator = (const PrivatePool &) = delete;
    // Number of live graphs using this pool
    int use_count{ 1 };
    // Number of unfreed npuMallocs made for this pool. When use_count and
    // npuMalloc_count drop to zero, we can delete this PrivatePool from
    // graph_pools.
    int npuMalloc_count{ 0 };
    // Instead of maintaining private BlockPools here, I could stuff all blocks
    // (private or no) into the top-level large_blocks and small_blocks, and
    // distinguish private blocks by adding a "pool id" check above the stream
    // check in BlockComparator. BlockComparator is performance- critical though,
    // I'd rather not add more logic to it.
    BlockPool large_blocks;
    BlockPool small_blocks;
};

BlockState::BlockState(Block* block)
    : device(block->device),
      stream(block->stream),
      stream_uses(block->stream_uses),
      size(block->size),
      ptr(block->ptr),
      allocated(block->allocated),
      gc_count_base(block->gc_count) {
  TORCH_CHECK(
      block->event_count == 0,
      "Events should have synchronized when checkpointing block", PTA_ERROR(ErrCode::INTERNAL));
};

SegmentState::SegmentState(Block* head)
{
    TORCH_INTERNAL_ASSERT(head != nullptr, PTA_ERROR(ErrCode::PTR));
    TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
    is_small = head->pool->is_small;

    for (Block* curr = head; curr != nullptr; curr = curr->next) {
        blocks.emplace_back(curr);
    }
}

PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Block*>& private_pool_head_blocks)
    : owner_id(std::move(pool_id))
{
    for (Block* head : private_pool_head_blocks) {
        segments.emplace_back(head);
    }
}

struct MempoolIdHash {
    std::size_t operator () (const MempoolId_t &mempool_id) const noexcept
    {
        return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
    }
};
} // namespace

bool checkConfigExpandableSegments()
{
    return CachingAllocatorConfig::expandable_segments();
}

bool isConfig1GPageSizeEnable()
{
    return CachingAllocatorConfig::page_size_1g_enable();
}

void setAllocatorSettings(const std::string& settings)
{
    ASCEND_LOGI("setAllocatorSettings: %s.", settings.c_str());
    // Empty NPU task queue before changing the allocator settings.
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    TORCH_CHECK(ret == NPU_STATUS_SUCCESS, "Failed to empty NPU task queue, ret:", ret, PTA_ERROR(ErrCode::INTERNAL));
    // Only support expandable_segments setting.
    CachingAllocatorConfig::instance().parseArgs(settings.c_str(), {"expandable_segments"});
}

bool saveDevMemUsageInfo(const int& device)
{
    aclrtMemUsageInfo memUsageInfo[MAX_MODULE_NUM] = {0};
    size_t moduleCount = 0;

    // Get the memory usage information
    aclError ret = c10_npu::acl::AclrtGetMemUsageInfo(device, memUsageInfo, MAX_MODULE_NUM, &moduleCount);
    if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGE("AclrtGetMemUsageInfo failed, ret:%d", ret);
        return false;
    }

    // Save the memory usage information to csv file
    std::time_t current_time = std::time(nullptr);
    std::tm *local_time = std::localtime(&current_time);
    std::ostringstream file_name_stream;
    file_name_stream << "device_" << device << "_memUsageInfo_" << std::put_time(local_time, "%Y%m%d%H%M%S") << ".csv";

    auto dumppath = c10_npu::option::OptionsManager::GetOomSnapshotDumpPath();
    auto savefilepath = c10::str(dumppath, "/", file_name_stream.str());

    std::ofstream csv_file(savefilepath);
    if (!csv_file.is_open()) {
        ASCEND_LOGE("Failed to open file: %s", savefilepath.c_str());
        return false;
    }

    csv_file << "moduleName,curMemSize(MB),memPeakSize(MB)\n" << std::fixed << std::setprecision(kPrecision);

    // moduleCount is unreliable, so limit i to MAX_MODULE_NUM
    for (size_t i = 0; i < moduleCount && i < MAX_MODULE_NUM; ++i) {
        csv_file << memUsageInfo[i].name << "," << static_cast<double>(memUsageInfo[i].curMemSize) / kMB << ","
                 << static_cast<double>(memUsageInfo[i].memPeakSize) / kMB << "\n";
    }
    if (moduleCount > MAX_MODULE_NUM) {
        ASCEND_LOGW("The number of modules exceeds the maximum limit: %zu > %zu", moduleCount, MAX_MODULE_NUM);
    }
    csv_file.close();

    ASCEND_LOGI("Device memory usage information saved to file: %s", savefilepath.c_str());
    return true;
}

// To prevent the deadlock situation, temporarily release the lock.
//
// Deadlock Scenario Description:
//
// 1. Main Thread:
//    - Acquires the lock and performs sync to clear the taskqueue.
//    - taskqueue wait a empty signal from the sub-thread.
//
// 2. Sub-thread:
//    - Python function (tbe op compile) called in CANN may trigger GC that introduces a resource release operation.
//    - The release operation (`free`) cannot acquire the same lock holded in main thread.
//    - Unable to send a signal to the main thread.
class UnlockGuard {
public:
    explicit UnlockGuard(std::unique_lock<std::recursive_mutex>& lock) : lock_(lock) { lock_.unlock(); }

    ~UnlockGuard() { lock_.lock(); }

private:
    std::unique_lock<std::recursive_mutex>& lock_;
};

struct handle_str {
    char data[ACL_IPC_HANDLE_SIZE];
};

// handle for ptr
ska::flat_hash_map<void *, handle_str> ipc_handle_map;

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
    ska::flat_hash_set<Block *> active_blocks;

    // captures_underway tracks if we are diverting some
    // allocations to a specific pool.
    // Most of the time it's empty, in which case malloc can avoid calling
    // aclrtStreamGetCaptureInfo in the hot path.
    std::vector<std::pair<MempoolId_t, std::function<bool(aclrtStream)>>> captures_underway;

    // See free() for this thing's purpose
    std::vector<Block *> needs_events_deferred_until_no_capture;

    // outstanding acl events
    ska::flat_hash_map<c10_npu::NPUStream, std::deque<std::pair<EventPool::Event, Block *>>> npu_events;

    // record used memory.
    size_t total_allocated_memory = 0;

    // record maximum allowed memory.
    size_t allowed_memory_maximum = 0;

    // all live expandable segments
    std::vector<ExpandableSegment *> expandable_segments_;

    bool set_fraction = false;

    bool record_history = false;

    std::atomic<CreateContextFn> context_recorder_;
    size_t alloc_trace_next = 0;
    RecordContext record_context_ = RecordContext::NEVER;
    size_t alloc_trace_max_entries_ = 1;
    std::vector<TraceEntry> *alloc_trace; // pointer because we need to intentionally leak this on
                                          // deallocation it can hold references to Python state which
                                          // will already be destroyed when we are in exit handlers

    // XXX - maybe we should generalize and have multiple events
    std::vector<OutOfMemoryObserver> oom_observers_;
    std::shared_ptr<c10d_npu::HCCLComm> hcclComm_;

    // Private pools for NPU graphs
    ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash> graph_pools;

    // Pools no longer referenced by any graph. Their BlockPools are eligible for
    // free_blocks. Can't be a vector or deque because we might erase entries in
    // any order. Could be an std::list, but we don't care much, access and
    // insert/erase are rare.
    ska::flat_hash_map<MempoolId_t, PrivatePool *, MempoolIdHash> graph_pools_freeable;

    // mapping from block to a stream_set, containing streams on which the block
    // was used while npugraph capturing
    std::unordered_map<Block *, stream_set> block_to_npugraph_stream_uses;

public:
    DeviceCachingAllocator() : large_blocks(false), small_blocks(true), alloc_trace(new std::vector<TraceEntry>())
    {
        stats.max_split_size = static_cast<int64_t>(CachingAllocatorConfig::max_split_size());
        context_recorder_.store(nullptr);
    }

    void recordHistory(bool enabled, CreateContextFn context_recorder, size_t alloc_trace_max_entries,
        RecordContext when)
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

    bool isHistoryEnabled()
    {
        return record_history;
    }

  bool checkPoolLiveAllocations(
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations)
  {
      std::unique_lock<std::recursive_mutex> lock(mutex);

      PrivatePool* pool = nullptr;
      auto pool_it = graph_pools.find(mempool_id);
      TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id", PTA_ERROR(ErrCode::INTERNAL));
      pool = pool_it->second.get();

      TORCH_INTERNAL_ASSERT(pool != nullptr, PTA_ERROR(ErrCode::PTR));
      size_t allocated_pool_blocks = 0;

      for (Block* b : active_blocks) {
          TORCH_INTERNAL_ASSERT(b != nullptr);
          TORCH_INTERNAL_ASSERT(b->pool != nullptr);
          if (b->allocated && b->pool->owner_PrivatePool == pool) {
              if (!expected_live_allocations.count(b->ptr)) {
                  return false;
              }
              allocated_pool_blocks += 1;
            }
      }

      return allocated_pool_blocks == expected_live_allocations.size();
  }

    void attachOutOfMemoryObserver(OutOfMemoryObserver observer)
    {
        oom_observers_.emplace_back(observer);
    }

    bool checkUceInMemPool()
    {
        auto memUceInfo_ = c10_npu::get_mem_uce_info();
        auto info = memUceInfo_.info;
        const auto all_blocks = get_all_blocks();
        bool any_found = false;
        aclrtMemUceInfo temp_info[memUceInfo_.retSize];
        size_t temp_retsize = 0;

        for (size_t i = 0; i < memUceInfo_.retSize; ++i) {
            void *addr = info[i].addr;
            size_t length = info[i].len;
            bool found = false;

            // Calculate the start and end address for info[i]
            void *addr_end = static_cast<char *>(addr) + length - 1;

            // Iterate through all blocks and check if there's an overlap with addr
            for (const Block * const head_block : all_blocks) {
                void *block_start = head_block->ptr;
                void *block_end = static_cast<char *>(head_block->ptr) + head_block->size - 1;

                // If there is an overlap, mark the block as unsafe
                if (addr <= block_end && addr_end >= block_start) {
                    const_cast<Block *>(head_block)->is_safe = false;
                    ASCEND_LOGI(
                        "Memory block with UCE fault error found in the NPUCachingAllocator and was marked as unsafe");
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
        for (auto &active_block : active_blocks) {
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

    Block *malloc(int device, size_t orig_size, aclrtStream stream, uint8_t allocator_type = 0)
    {
        // done outside the lock because we don't know what locks the recorder needs
        // to have...
        auto context = maybeGatherContext(RecordContext::STATE);

        std::unique_lock<std::recursive_mutex> lock(mutex);

        if (device == -1) {
            NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        }

        if (C10_LIKELY(captures_underway.empty())) {
            // Processes end-of-life events for outstanding allocations used on
            // multiple streams (checks if their NPU-side uses are complete and
            // recycles their memory if so)
            //
            // Q. Why skip process_events if a capture might be underway?
            // A. process_events involves npuEventQueries, illegal during NPU graph
            //    capture.
            //    Dumb simple solution: defer reclaiming these allocations until after
            //    capture. Cross-stream memory use is uncommon, so the deferral's
            //    effect on memory use during capture should be small.
            process_events(context);
        }
        auto size = round_size(orig_size);
        auto &pool = get_pool(size, stream);

        // 开环境变量 大池子放1G内存块
        const size_t alloc_size = IsMallocPage1GMem(pool.is_small) ?
            kExtraLargeBuffer * ((size + kExtraLargeBuffer - 1) / kExtraLargeBuffer) :
            get_allocation_size(size);
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
            if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
                garbage_collect_cached_blocks(context, lock);
            }
            // Attempt allocate
            block_found = alloc_block(params, false, context, lock) ||
                // Free enough available cached blocks to satisfy alloc and retry
                // alloc.
                (release_available_cached_blocks(params, context, lock) && alloc_block(params, false, context, lock));
        }

        if (!block_found && C10_LIKELY(captures_underway.empty())) {
            ASCEND_LOGE(
                "Get a block from the existing pool failed. Try to free cached blocks and reallocate. This error log can be ignored.");
            // Free all non-split cached blocks and retry alloc.
            {
                UnlockGuard guard(lock);
                // Make sure taskqueue is empty, then execute release_cached_blocks
                c10_npu::npuSynchronizeDevice(true);
            }
            c10_npu::NPUWorkspaceAllocator::emptyCache(device, true);
            block_found = (release_cached_blocks(true, context, true) && alloc_block(params, true, context, lock));
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

                record_trace(TraceEntry::OOM, device_free, params.size(), params.stream(), params.device(),
                    std::move(context));
                auto observers_local = oom_observers_;

                // Make sure we do not have the device lock before calling our
                // observers which might need hold the GIL
                // It is safe to release at this point because will no longer
                // be reading any allocator state.

                lock.unlock();

                for (const auto &obs : observers_local) {
                    obs(device, alloc_size, set_fraction ? allowed_memory_maximum : device_total, device_free);
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
                auto retmsg = std::string("NPU out of memory. Tried to allocate ") + format_size(alloc_size) +
                    " (NPU " + std::to_string(device) + "; " + format_size(device_total) + " total capacity; " +
                    format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current) +
                    " already allocated; " +
                    format_size(stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current) +
                    " current active; " + format_size(device_free) + " free; " + allowed_info +
                    format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current) +
                    " reserved in total by PyTorch)." +
                    "If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.";

                ASCEND_LOGE("%s", retmsg.c_str());
                TORCH_CHECK_WITH(OutOfMemoryError, false, retmsg.c_str());
            } else {
                NPU_CHECK_ERROR(params.err);
            }
        }

        int64_t ori_block_ptr = int64_t(params.block->ptr);
        size_t align_round = CachingAllocatorConfig::base_addr_aligned_size();
        if (params.size() >= kRoundLarge && CachingAllocatorConfig::expandable_segments() && align_round != 0 &&
            ori_block_ptr % align_round != 0) {
            char *align_ptr = reinterpret_cast<char *>((ori_block_ptr + align_round) - (ori_block_ptr % align_round));
            size_t offset_size = align_ptr - (char *)params.block->ptr;
            if (offset_size + params.size() <= params.block->size) {
                auto size = params.block->size;
                Block *remaining = params.block;

                Block *block = new Block(params.device(), params.stream(), size - offset_size, params.pool, align_ptr);
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
        return alloc_found_block(std::move(params), orig_size, std::move(context), split_remainder, allocator_type);
    }

    Block *alloc_found_block(AllocParams params, size_t orig_size, std::shared_ptr<c10::GatheredContext> context,
        bool split_remainder, uint8_t allocator_type)
    {
        auto size = params.size();
        auto device = params.device();
        auto pool = params.pool;
        auto stream = params.stream();

        TORCH_INTERNAL_ASSERT(params.err == ACL_ERROR_NONE && params.block != nullptr && params.block->ptr != nullptr,
            PTA_ERROR(ErrCode::PTR));
        Block *block = params.block;
        Block *remaining = nullptr;

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
            remaining->ptr = static_cast<char *>(remaining->ptr) + size;
            remaining->size -= size;
            pool->blocks.insert(remaining);

            if (already_split && !block->expandable_segment_) {
                // An already-split inactive block is being shrunk by size bytes.
                update_stat_array(stats.inactive_split_bytes, -static_cast<std::int64_t>(block->size),
                    params.stat_types);
            } else if (!block->expandable_segment_) {
                // A new split inactive block is being created from a previously unsplit
                // block, size remaining->size bytes.
                for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
                    update_stat(stats.inactive_split_bytes[stat_type], static_cast<std::int64_t>(remaining->size));
                    update_stat(stats.inactive_split[stat_type], 1);
                });
            }
        } else if (already_split && !block->expandable_segment_) {
            // An already-split block is becoming active
            for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
                update_stat(stats.inactive_split_bytes[stat_type], -static_cast<std::int64_t>(block->size));
                update_stat(stats.inactive_split[stat_type], -1);
            });
        }

        block->allocated = true;
        block->requested_size = orig_size;
        if (block->is_safe == false) {
            ASCEND_LOGI("Unsafe memory block is passively refreshed by releasing and mallocing memory again");
        }
        block->is_safe = true;

        block->context_when_allocated = std::move(context);
        record_trace(TraceEntry::ALLOC, int64_t(block->ptr), orig_size, block->stream, block->device,
            block->context_when_allocated);

        active_blocks.insert(block);

        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
            update_stat(stats.allocation[stat_type], 1);
            update_stat(stats.allocated_bytes[stat_type], static_cast<std::int64_t>(block->size));
            update_stat(stats.active[stat_type], 1);
            update_stat(stats.active_bytes[stat_type], static_cast<std::int64_t>(block->size));
            update_stat(stats.requested_bytes[stat_type], static_cast<std::int64_t>(block->requested_size));
        });

        if (block->size >= CachingAllocatorConfig::max_split_size()) {
            update_stat(stats.oversize_allocations, 1);
        }

        ASCEND_LOGD("PTA CachingAllocator malloc: malloc = %zu, cached = %lu, allocated = %lu", block->size,
            stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);

#ifndef BUILD_LIBTORCH
        if (torch_npu::profiler::MstxMgr::GetInstance()->isMsleaksEnable()) {
            mstxDomainHandle_t msleaksDomain = torch_npu::profiler::MstxMgr::GetInstance()->createLeaksDomain(
                torch_npu::profiler::DOMAIN_CACHING.c_str());
            mstxMemVirtualRangeDesc_t heapDesc{ block->device, block->ptr,
                stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current };
            torch_npu::profiler::MstxMgr::GetInstance()->memHeapRegister(msleaksDomain, &heapDesc);
            mstxMemVirtualRangeDesc_t regionDesc{ block->device, block->ptr, block->size };
            torch_npu::profiler::MstxMgr::GetInstance()->memRegionsRegister(msleaksDomain, &regionDesc);
        }
        torch_npu::profiler::reportMemoryDataToNpuProfiler({ static_cast<int8_t>(c10::DeviceType::PrivateUse1),
            block->device, static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::CACHING_ALLOCATOR),
            static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_MALLOC), allocator_type,
            reinterpret_cast<int64_t>(block->ptr), block->size,
            stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            block->stream });
#endif

        return block;
    }


    void free(Block *block, uint8_t allocator_type = 0)
    {
        std::shared_ptr<c10::GatheredContext> context = maybeGatherContext(RecordContext::ALL);
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

        record_trace(TraceEntry::FREE_REQUESTED, int64_t(block->ptr), block->requested_size, block->stream,
            block->device, context ? context : block->context_when_allocated);

        if (block->size >= CachingAllocatorConfig::max_split_size()) {
            update_stat(stats.oversize_allocations, -1);
        }

        if (!block->stream_uses.empty() && c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
            if (C10_UNLIKELY(!captures_underway.empty())) {
                // It's forbidden to npuEventQuery an event recorded during NPU graph
                // capture. We conservatively defer recording end-of-life events until
                // the next call to process_events() (which won't happen until no
                // captures are underway)
                needs_events_deferred_until_no_capture.push_back(block);
            } else {
                insert_events(block);
            }
        } else {
            free_block(block, context, allocator_type);
        }

        ASCEND_LOGD("PTA CachingAllocator free: free = %zu, cached = %lu, allocated = %lu", orig_block_size,
            stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
#ifndef BUILD_LIBTORCH
        if (torch_npu::profiler::MstxMgr::GetInstance()->isMsleaksEnable()) {
            mstxDomainHandle_t msleaksDomain = torch_npu::profiler::MstxMgr::GetInstance()->createLeaksDomain(
                torch_npu::profiler::DOMAIN_CACHING.c_str());
            mstxMemVirtualRangeDesc_t desc{ block->device, orig_block_ptr,
                stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current };
            torch_npu::profiler::MstxMgr::GetInstance()->memHeapRegister(msleaksDomain, &desc);
            torch_npu::profiler::MstxMgr::GetInstance()->memRegionsUnregister(msleaksDomain, orig_block_ptr);
        }
        torch_npu::profiler::reportMemoryDataToNpuProfiler({ static_cast<int8_t>(c10::DeviceType::PrivateUse1),
            block->device, static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::CACHING_ALLOCATOR),
            static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_FREE), allocator_type,
            reinterpret_cast<int64_t>(orig_block_ptr), -orig_block_size,
            stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            block->stream });
#endif
    }

    void *getBaseAllocation(Block *block, size_t *outSize)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        while (block->prev) {
            block = block->prev;
        }
        void *basePtr = block->ptr;
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

    ShareableHandle shareIpcHandle(Block* block)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        std::ostringstream ss;
        ss.put(SHAREABLE_HANDLE_VERSION);
        ptrdiff_t offset = 0;
        if (!block->expandable_segment_) {
            ss.put(SHAREABLE_NPU_MALLOC);
            size_t base_size;
            void* base_ptr = getBaseAllocation(block, &base_size);
            offset = (char*)block->ptr - (char*)base_ptr;

            handle_str handle;
            auto it = ipc_handle_map.find(base_ptr);
            if (it == ipc_handle_map.end()) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtIpcMemGetExportKey(
                    base_ptr, base_size, handle.data, ACL_IPC_HANDLE_SIZE, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION));
                ipc_handle_map[base_ptr] = handle;
            } else {
                handle = it->second;
            }
            ss.write((char*)&handle, ACL_IPC_HANDLE_SIZE);
        } else {
            ss.put(SHAREABLE_NPU_EXPANDABLE_SEGMENT);
            auto full_range = block->expandable_segment_->share(
                SegmentRange(block->ptr, block->size), ss);
            offset = (char*)block->ptr - (char*)full_range.ptr;
        }
        return ShareableHandle{offset, ss.str()};
    }

    void recordStream(Block *block, c10_npu::NPUStream stream)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        block->stream_uses.insert(stream);
        if (C10_UNLIKELY(!captures_underway.empty())) {
            block_to_npugraph_stream_uses[block].insert(stream);
        }
    }

    void eraseStream(Block *block, c10_npu::NPUStream stream)
    {
        std::shared_ptr<c10::GatheredContext> context = maybeGatherContext(RecordContext::ALL);
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

    /* * set memory fraction to limit maximum allocated memory * */
    void setMemoryFraction(double fraction)
    {
        size_t device_free;
        size_t device_total;
        NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
        allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
        set_fraction = true;
    }

    /* * returns cached blocks to the system allocator * */
    void emptyCache(int device, bool check_error, bool free_physical)
    {
        std::shared_ptr<c10::GatheredContext> context = maybeGatherContext(RecordContext::ALL);
        // Make sure event deque from taskqueue, then synchronize Event
        c10_npu::npuSynchronizeDevice(check_error);
        std::lock_guard<std::recursive_mutex> lock(mutex);
        c10_npu::NPUWorkspaceAllocator::emptyCache(device, check_error);
        release_cached_blocks(check_error, context, free_physical);
    }

    void buildServerMemMapForHccl(std::shared_ptr<c10d_npu::HCCLComm> hcclComm)
    {
        std::unique_lock<std::recursive_mutex> lock(mutex);
        TORCH_INTERNAL_ASSERT(!hcclComm_, "Build HCCL server group redundancy.", PTA_ERROR(ErrCode::INTERNAL));
        hcclComm_ = hcclComm;
        for (auto &expandable_segments : expandable_segments_) {
            expandable_segments->setHcclComm(hcclComm);
        }
    }

    void release_and_free_events()
    {
        std::unique_lock<std::recursive_mutex> lock(mutex);
        std::shared_ptr<c10::GatheredContext> context = maybeGatherContext(RecordContext::ALL);
        for (auto &st : npu_events) {
            for (auto &e : st.second) {
                EventPool::Event event = std::move(e.first);
                Block *block = e.second;
                block->event_count--;
                if (block->event_count == 0) {
                    free_block(block, context);
                }
            }
        }
        npu_events.clear();
    }

    /* * Retrieves info (total size + largest block) of the memory cache * */
    void cacheInfo(size_t *total, size_t *largest)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        cache_info_aux(large_blocks, total, largest);
        cache_info_aux(small_blocks, total, largest);
        for (const auto &gp : graph_pools) {
            cache_info_aux(gp.second->large_blocks, total, largest);
            cache_info_aux(gp.second->small_blocks, total, largest);
        }
    }

    /* * Returns a copy of the memory allocator stats * */
    DeviceStats getStats()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        return stats;
    }

    /* * Resets the historical accumulation stats for the device * */
    void resetAccumulatedStats()
    {
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

    /* * Resets the historical peak stats for the device * */
    void resetPeakStats()
    {
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

    /* Checkpoint the state of a private pool necessary to return it to its
      * current state */
    std::unique_ptr<PrivatePoolState> getCheckpointState(MempoolId_t id)
    {
        auto context = maybeGatherContext(RecordContext::ALL);
        std::lock_guard<std::recursive_mutex> lock(mutex);
        insert_events_deferred_until_no_capture(context);

        auto pool = graph_pools.find(id);
        if (pool != graph_pools.end()) {
            auto private_pool_head_blocks = get_private_pool_head_blocks(pool->second.get());
            return std::make_unique<PrivatePoolState>(id, private_pool_head_blocks);
        } else if (graph_pools_freeable.count(id)) {
            TORCH_CHECK(false, "Not expected to checkpoint freeable graph", PTA_ERROR(ErrCode::VALUE));
        } else {
            TORCH_CHECK(false, "Could not find pool of id", PTA_ERROR(ErrCode::NOT_FOUND));
        }
    }

    void freeBlocksAllocatedToPool(PrivatePool* private_pool, RestoreResult& rr)
    {
        auto pool_blocks = get_private_pool_head_blocks(private_pool);

        std::vector<Block*> head_blocks;
        for (Block* block : pool_blocks) {
            if (block->prev == nullptr) {
                head_blocks.push_back(block);
            }
        }
        for (Block* block : head_blocks) {
            Block* curr = block;

            while (curr) {
                // When we free a block, its pointer should never change
                // only its adjacent blocks, so free, then look at pointer
                if (curr->allocated) {
                    TORCH_CHECK(
                        curr->event_count == 0,
                        "Events should have synchronized when setting checkpointed block", PTA_ERROR(ErrCode::INTERNAL));
                    rr.allocations_freed.push_back(curr->ptr);
                    free(curr);
                    TORCH_CHECK(!curr->allocated, PTA_ERROR(ErrCode::PTR));
                }
                curr = curr->next;
            }
        }
        for (Block* b : get_private_pool_head_blocks(private_pool)) {
            Block* curr = b;
            while (curr) {
                TORCH_CHECK(!curr->allocated, PTA_ERROR(ErrCode::PTR));
                curr = curr->next;
            }
        }
    }

    // checkpoint the state of an allocation that may have been
    // split into multiple blocks
    void setSegmentStateToCheckpoint(
        Block* block,
        SegmentState& segment,
        const std::shared_ptr<c10::GatheredContext>& context,
        RestoreResult& rr)
    {
        Block* curr_block = block;
        Block* last_block = block;

        TORCH_INTERNAL_ASSERT(block->pool);
        BlockPool& pool = *block->pool;
        const auto segment_len = segment.blocks.size();

        // allocate all blocks in the segment
        for (size_t i = 0; i < segment_len; ++i) {
            // The last block in every expandable segment is the remaining amount of
            // available unmapped virtual address space. We shouldn't change it but
            // instead check it is correctly formed then skip over allocating it.
            if (i == segment_len - 1 && curr_block->expandable_segment_) {
                // In case where expandable_segment is enabled, memory blocks will be merged when they are released.
                // Therefore, when a small memory block is allocated first, followed by a larger memory block,
                // and both are subsequently freed. If we attempt to restore the segment state by check point
                // of the allocated small memory block, we will observe that the next memory block of the last block
                // is not nullptr, and the last block is also a mapped block.
                // This is reasonable because blocks are merged. Hence, we will remove those excessive validations.
                // For more details, see https://github.com/pytorch/pytorch/issues/161356.
                TORCH_CHECK(curr_block->allocated == false, PTA_ERROR(ErrCode::VALUE));
                continue;
            }

            auto& block_state = segment.blocks.at(i);
            AllocParams params(
                block_state.device,
                block_state.size,
                block_state.stream,
                &pool,
                block_state.size,
                stats);
            pool.blocks.erase(curr_block);
            params.block = curr_block;
            params.stat_types = get_stat_types_for_pool(pool);

            // splitting a block depends on `max_split_size`, which may have changed
            // between when checkpoint was taken and now, so we make sure to recreate
            // the behavior from the checkpoint. Keep splitting as long as there is
            // space left in the block because the block is already the size of how it
            // appears in the segment, so any leftover space belongs to the next
            // block.
            bool split = curr_block->size > block_state.size;

            // curr_block will become next pointer if it is split, so reassign with
            // the returned value
            curr_block = alloc_found_block(params, block_state.size, context, split, 0);

            TORCH_CHECK(curr_block->ptr == block_state.ptr, PTA_ERROR(ErrCode::PTR));
            TORCH_CHECK(curr_block->size == block_state.size, PTA_ERROR(ErrCode::VALUE));

            last_block = curr_block;
            curr_block = curr_block->next;

            TORCH_CHECK((curr_block != nullptr) == ((i + 1) < (segment_len)), PTA_ERROR(ErrCode::INTERNAL));
        }

        while (last_block->prev) {
            last_block = last_block->prev;
        }

        // free blocks that are not allocated in the checkpoint
        curr_block = last_block;

        for (size_t i = 0; i < segment_len; ++i, curr_block = curr_block->next) {
            if (i == segment_len - 1 && curr_block->expandable_segment_) {
                // The same reason as above.
                TORCH_CHECK(curr_block->allocated == false, PTA_ERROR(ErrCode::VALUE));
                continue;
            }

            auto& block_state = segment.blocks.at(i);
            TORCH_INTERNAL_ASSERT(curr_block != nullptr, PTA_ERROR(ErrCode::PTR));

            if (block_state.allocated) {
                rr.allocations_created.push_back(curr_block);
                continue;
            }

            free(curr_block);

            TORCH_CHECK(curr_block->ptr == block_state.ptr, PTA_ERROR(ErrCode::PTR));
            TORCH_CHECK(curr_block->allocated == block_state.allocated, PTA_ERROR(ErrCode::VALUE));
            if (!curr_block->expandable_segment_) {
                // In case where expandable_segment is enabled, memory blocks will be merged when they are released.
                // The size of curr_block may be greater than the size of block_state.
                // Therefore the block size assertion is also excessive in expandable_segment.
                TORCH_CHECK(curr_block->size == block_state.size, PTA_ERROR(ErrCode::VALUE));
            }
        }
    }
    /**
     * Note [Checkpointing PrivatePoolState]
     *
     * Refer above to Note [Interaction with acl graph capture]. Allocations made
     * during graph capture are made from a separate private pool. During graph
     * capture allocations behave as usual. During graph replay the allocator
     * state does not change even as new tensors are created. The private pool
     * will not free its blocks to the main caching allocator until acl graph use
     * is finished to prevent an allocation from eager clobbering the memory from
     * a live but unaccounted for tensor that was created during replay.
     *
     * `make_graphed_callables`, a series of separate callables chained in
     * successive acl graphs, can share a memory pool because after a acl graph
     * recording the allocations in the shared private pool exactly reflect the
     * tensors that are allocated.
     *
     * We would like to extend callable chaining to support a graphed callable
     * tree. In this scenario, we have a tree of callable chains which will be
     * captured with acl graphs. In the diagram below, we have a tree with four
     * callables, A, B, C, and D. Suppose we have captured, and subsequently
     * replayed, A, B, and C. Then on a new invocation, we replay A and B, but
     * would now like to record D. At this point the private pool will not reflect
     * any of the live tensors created during graph replay. Allocations made
     * during a new recording with the pool could overwrite those live tensors.
     *
     * In order to record a new graph capture after replaying prior callables in
     * the tree, we need the allocator to reflect the state of the live tensors.
     * We checkpoint the state of the private pool after each recording, and then
     * reapply it when we are starting a new recording chain. Additionally, we
     * must free the allocations for any tensors that died between the end of our
     * previous graph replaying and our new recording. All of the allocated
     * segments that existed in the checkpointed state must still exist in the
     * pool. There may also exist new allocated blocks.
     * (link note [live tensors between iterations] when it exists). For
     * every block that is currently allocated but no allocated in the snapshot,
     * we will return a pointer to their block.
     *.
    *
    *
    *  ---------------> A ---------------> B ---------------> C
    *                                      |
    *                                      |
    *                                      |
    *                                      |
    *                                      ╰ ---------------> D
    */
    RestoreResult setCheckpointPoolState(PrivatePoolState& pps)
    {
        // To reset the caching allocator state we will
        // - Free all the blocks currently allocated to the pool (see [live tensors
        // between iterations])
        // - Allocate all the blocks in a checkpointed segment, whether they are
        // live or not
        // - Free the blocks in a checkpointed segment which are not live
        // This could be optimized, but it nicely reuses exiting apis, and this
        // is not on the hot path.

        // following `done outside the lock because we don't know what locks the
        // recorder needs to have...`

        std::shared_ptr<c10::GatheredContext> context =
            maybeGatherContext(RecordContext::STATE);

        std::lock_guard<std::recursive_mutex> lock(mutex);

        RestoreResult rr;

        TORCH_CHECK(
            !graph_pools_freeable.count(pps.owner_id),
            "Not expected to checkpoint freeable graph", PTA_ERROR(ErrCode::INTERNAL));

        auto pool = graph_pools.find(pps.owner_id);
        TORCH_CHECK(pool != graph_pools.end(), "Could not find private pool id", PTA_ERROR(ErrCode::INTERNAL));

        PrivatePool* private_pool = pool->second.get();

        freeBlocksAllocatedToPool(private_pool, rr);

        std::unordered_map<void*, Block*> ptrs_to_blocks;
        // at this point, all of the blocks should be free, so they will all be in
        // the block set
        for (Block* block : private_pool->small_blocks.blocks) {
            ptrs_to_blocks[block->ptr] = block;
        }
        for (Block* block : private_pool->large_blocks.blocks) {
            ptrs_to_blocks[block->ptr] = block;
        }
        for (auto& segment : pps.segments) {
            auto ptr = segment.blocks.at(0).ptr;
            TORCH_CHECK(ptrs_to_blocks.count(ptr), " could not find ", ptr, PTA_ERROR(ErrCode::PARAM));
            auto block = ptrs_to_blocks[ptr];

            setSegmentStateToCheckpoint(block, segment, context, rr);
        }
        return rr;
    }

    /* * Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. * */
    std::vector<SegmentInfo> snapshot()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);

        std::unordered_map<PrivatePool *, MempoolId_t> pool_to_id;
        pool_to_id.reserve(graph_pools.size() + graph_pools_freeable.size());
        for (const auto &pair : graph_pools) {
            pool_to_id[pair.second.get()] = pair.first;
        }
        for (const auto &pair : graph_pools_freeable) {
            pool_to_id[pair.second] = pair.first;
        }

        uint64_t total_active = 0;
        std::vector<SegmentInfo> result;
        const auto all_blocks = get_all_blocks();

        for (const Block * const head_block : all_blocks) {
            // For expandable segments, we report one segment for each continguous
            // mapped range of memory
            if (head_block->prev && head_block->prev->mapped) {
                continue;
            }
            result.emplace_back();
            SegmentInfo &segment_info = result.back();
            segment_info.device = head_block->device;
            segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
            segment_info.stream = head_block->stream;
            segment_info.is_large = (!head_block->pool->is_small);
            segment_info.is_expandable = head_block->expandable_segment_;
            segment_info.context_when_allocated = head_block->context_when_segment_allocated;
            auto mempool_id = pool_to_id.find(head_block->pool->owner_PrivatePool);
            if (mempool_id != pool_to_id.end()) {
                segment_info.owner_private_pool_id = mempool_id->second;
            }
            const Block *block = head_block;
            while (block != nullptr && block->mapped) {
                segment_info.blocks.emplace_back();
                BlockInfo &block_info = segment_info.blocks.back();

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
            [](const SegmentInfo &a, const SegmentInfo &b) { return a.address < b.address; });

        record_trace(TraceEntry::SNAPSHOT, 0, total_active, nullptr, 0, nullptr);
        return result;
    }

    std::vector<TraceEntry> trace()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        std::vector<TraceEntry> result;
        result.reserve(alloc_trace->size());
        result.insert(result.end(), alloc_trace->begin() + alloc_trace_next, alloc_trace->end());
        result.insert(result.end(), alloc_trace->begin(), alloc_trace->begin() + alloc_trace_next);

        return result;
    }

    // This function takes the size and number of divisions argument and rounds
    // up the size argument for the nearest power-of-2 division.
    // For example, if we need to round-up 1200 and number of divisions is 4,
    // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
    // them, the values are 1024, 1280, 1536, and 1792. So the function will
    // return 1280 as the nearest ceiling of power-2 division.
    static size_t roundup_power2_next_division(size_t size, size_t divisions)
    {
        if (c10::llvm::isPowerOf2_64(size)) {
            return size;
        }

        constexpr size_t kMinDivisions = 2;
        TORCH_CHECK(divisions >= kMinDivisions,
            "Only 2 or more divisions are supported", PTA_ERROR(ErrCode::NOT_SUPPORT));

        // divide the space between these 2's power into equal divisions
        // If division is zero, return the power-of-2 ceiling.
        size_t power2_floor = c10::llvm::PowerOf2Floor(size);
        size_t power2_division = power2_floor >> (63 - c10::llvm::countLeadingZeros(divisions));
        if (C10_UNLIKELY(power2_division == 0)) {
            return (power2_floor << 1);
        }
        size_t round_size_floor = size & (~(power2_division - 1));
        return (round_size_floor == size) ? size : round_size_floor + power2_division;
    }

    static size_t round_size(size_t size)
    {
        size += AddPadSize();
        if (size < kMinBlockSize) {
            return kMinBlockSize;
        } else {
            auto divisions = CachingAllocatorConfig::roundup_power2_divisions(size);
            if (divisions > 1 && size > (kMinBlockSize * divisions)) {
                return roundup_power2_next_division(size, divisions);
            } else {
                return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
            }
        }
    }

    // See Note [Interaction with NPU graph capture]

    // Called by NPUGraph::capture_begin
    void beginAllocateToPool(MempoolId_t mempool_id, std::function<bool(aclrtStream)> filter)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto it = graph_pools.find(mempool_id);
        if (it == graph_pools.end()) {
            // mempool_id does not reference an existing pool. Make a new pool for
            // this capture.
            graph_pools.emplace(mempool_id, std::make_unique<PrivatePool>());
        } else {
            // mempool_id references an existing pool, which the current capture will
            // share. Check this pool is live (at least one other capture already
            // references it).
            TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
            it->second->use_count++;
        }
        for (auto it2 = captures_underway.begin(); it2 != captures_underway.end(); ++it2) {
            TORCH_CHECK(it2->first != mempool_id, "beginAllocateToPool: already recording to mempool_id");
        }
        captures_underway.emplace_back(mempool_id, std::move(filter));
    }

    // Called by NPUGraph::capture_end
    void endAllocateToPool(MempoolId_t mempool_id)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (auto it = captures_underway.begin(); it != captures_underway.end(); ++it) {
            if (it->first == mempool_id) {
                captures_underway.erase(it);
                return;
            }
        }
        TORCH_CHECK(false, "endAllocatePool: not currently recording to mempool_id");
    }

    // Called by NPUGraph::reset
    void releasePool(MempoolId_t mempool_id)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        // The instantiated npugraphExec_t has been destroyed. We can't blindly
        // delete and npuFree the mempool its capture used, because
        //  1. other graph(s) might share the same pool
        //  2. the user might still hold references to output tensors allocated
        //  during capture.
        // To handle 1 and 2, we track the number of graphs using this particular
        // mempool. When the count reaches 0, we tell free_cached_blocks it may now
        // npuFree blocks from this graph's pool when it discovers they're unused
        // (unsplit).
        auto it = graph_pools.find(mempool_id);
        TORCH_INTERNAL_ASSERT(it != graph_pools.end());
        auto uc = --(it->second->use_count);
        TORCH_INTERNAL_ASSERT(uc >= 0);
        if (uc == 0) {
            // Allows free_cached_blocks to begin npuFreeing this pool's memory,
            // and makes sure this pool wasn't somehow made freeable already.
            // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
            bool inserted = graph_pools_freeable.insert({ mempool_id, it->second.get() }).second;
            TORCH_INTERNAL_ASSERT(inserted);
        }
    }

private:
    // All private methods do not acquire the allocator mutex.

    std::vector<const Block *> get_all_blocks() const
    {
        std::vector<const Block *> blocks;
        blocks.insert(blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
        blocks.insert(blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
        for (const auto &gp : graph_pools) {
            blocks.insert(blocks.end(), gp.second->small_blocks.blocks.begin(), gp.second->small_blocks.blocks.end());
            blocks.insert(blocks.end(), gp.second->large_blocks.blocks.begin(), gp.second->large_blocks.blocks.end());
        }
        blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
        return blocks;
    }

    std::vector<Block*> get_private_pool_head_blocks(PrivatePool* pool) const
    {
        std::vector<Block*> blocks;
        for (Block* b : active_blocks) {
            if ((b->pool == &pool->small_blocks || b->pool == &pool->large_blocks) &&
                b->prev == nullptr) {
                blocks.push_back(b);
            }
        }
        for (Block* b : pool->small_blocks.blocks) {
            if (b->prev == nullptr) {
                blocks.push_back(b);
            }
        }
        for (Block* b : pool->large_blocks.blocks) {
            if (b->prev == nullptr) {
                blocks.push_back(b);
            }
        }

        return blocks;
    }

    // returns the smallest possible address in any segment
    // where there is enough free address space to fit size
    // may be composed of free and unmapped segments
    Block *find_expandable_block(int device, aclrtStream stream, BlockPool *pool, size_t size)
    {
        Block key(device, stream, 0);

        auto allocatable = [](Block *b) { return b && !b->allocated && b->event_count == 0 && b->stream_uses.empty(); };
        auto has_available_address_space = [&](Block *b) {
            size_t bytes = 0;
            while (bytes < size && allocatable(b)) {
                bytes += b->size;
                b = b->next;
            }
            return bytes >= size;
        };
        for (auto it = pool->unmapped.lower_bound(&key); it != pool->unmapped.end() && (*it)->stream == stream; ++it) {
            Block *c = *it;
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
        auto custom_segment_size = CachingAllocatorConfig::segment_size_mb();
        auto segment_size = pool->is_small ?
                kSmallBuffer : (custom_segment_size > 0 ? custom_segment_size : kLargeBuffer);
        // 此处申请虚拟内存，segment_size是页大小，实际虚拟内存巨大
        if (IsMallocPage1GMem(pool->is_small)) {
            segment_size = kExtraLargeBuffer;
        }
        auto segment = new (std::nothrow) ExpandableSegment(device, stream, segment_size);
        if (!segment) {
            ASCEND_LOGE("Failed to allocate ExpandableSegment.");
            return nullptr;
        }
        if (hcclComm_) {
            segment->setHcclComm(hcclComm_);
        }
        expandable_segments_.emplace_back(segment);

        ExpandableSegment *es = expandable_segments_.back();
        Block *candidate = new (std::nothrow) Block(device, stream, es->size(), pool, es->ptr());
        if (!candidate) {
            ASCEND_LOGE("Failed to allocate Block.");
            return nullptr;
        }
        candidate->mapped = false;
        candidate->expandable_segment_ = es;
        pool->unmapped.insert(candidate);
        return candidate;
    }

    bool map_block(Block *to_map, size_t size, const std::shared_ptr<c10::GatheredContext> &ctx, BlockPool *map_pool)
    {
        TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size, PTA_ERROR(ErrCode::VALUE));
        TORCH_INTERNAL_ASSERT(!to_map->context_when_allocated); // unmapped blocks should not keep
                                                                // history
        auto mapped_range = to_map->expandable_segment_->map(SegmentRange{ to_map->ptr, size }, map_pool);
        // failed to map the memory
        if (mapped_range.size == 0) {
            return false;
        }
        TORCH_INTERNAL_ASSERT(mapped_range.ptr == to_map->ptr && mapped_range.size >= size,
            PTA_ERROR(ErrCode::INTERNAL));

        BlockPool &pool = *to_map->pool;
        pool.unmapped.erase(to_map);
        to_map->mapped = true;

        if (mapped_range.size < to_map->size) {
            // to_map -> remaining -> to_map->next(?)
            Block *remaining = new Block(to_map->device, to_map->stream, to_map->size - mapped_range.size, &pool,
                static_cast<char *>(to_map->ptr) + mapped_range.size);
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
        for_each_selected_stat_type(stat_types,
            [&](size_t stat_type) { update_stat(stats.reserved_bytes[stat_type], mapped_range.size); });
        record_trace(TraceEntry::SEGMENT_MAP, int64_t(mapped_range.ptr), mapped_range.size, to_map->stream,
            to_map->device, ctx);
        if (!to_map->prev && !to_map->context_when_segment_allocated) {
            to_map->context_when_segment_allocated = ctx;
        }

        return true;
    }

    Block *try_allocate_expandable_block(int device, aclrtStream stream, BlockPool *pool, size_t size,
        const std::shared_ptr<c10::GatheredContext> &ctx)
    {
        Block *candidate = find_expandable_block(device, stream, pool, size);
        // Candidate is now a list free/unmapped blocks with at least size room:
        // unmapped -> null
        // unmapped -> free -> *
        // free -> unmapped -> *

        if (!candidate->mapped && !map_block(candidate, std::min(candidate->size, size), ctx, pool)) {
            return nullptr;
        }
        TORCH_INTERNAL_ASSERT(candidate->mapped, PTA_ERROR(ErrCode::INTERNAL));

        while (candidate->size < size) {
            // invariant: free -> unmapped -> *
            // map_block will map some of unmapped and merge with free
            auto remaining = size - candidate->size;
            auto new_candidate = candidate->next;
            if (C10_UNLIKELY(new_candidate == nullptr)) {
                return nullptr;
            }
            if (!map_block(new_candidate, std::min(remaining, candidate->next->size), ctx, pool)) {
                return nullptr;
            }
            candidate = new_candidate;
        }
        pool->blocks.erase(candidate);
        return candidate;
    }


    /* * moves a block into a pool of cached free blocks * */
    void free_block(Block *block, const std::shared_ptr<c10::GatheredContext> &context, uint8_t allocator_type = 0)
    {
        AT_ASSERT(!block->allocated && block->event_count == 0, PTA_ERROR(ErrCode::VALUE));

        record_trace(TraceEntry::FREE_COMPLETED, int64_t(block->ptr), block->requested_size, block->stream,
            block->device, context ? context : block->context_when_allocated);

        block->context_when_allocated = nullptr;
        block->hccl_work_ptr = nullptr;
        size_t original_block_size = block->size;
        auto orig_block_ptr = block->ptr;
        size_t requested_size = block->requested_size;

        auto &pool = *block->pool;
        int64_t net_change_inactive_split_blocks = 0;
        int64_t net_change_inactive_split_size = 0;

        const std::array<Block *, 2> merge_candidates = { block->prev, block->next };
        for (Block *merge_candidate : merge_candidates) {
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
                update_stat(stats.inactive_split[stat_type], net_change_inactive_split_blocks);
                update_stat(stats.inactive_split_bytes[stat_type], net_change_inactive_split_size);
            }
            update_stat(stats.active[stat_type], -1);
            update_stat(stats.active_bytes[stat_type], -original_block_size);
            update_stat(stats.requested_bytes[stat_type], -static_cast<std::int64_t>(requested_size));
        });
#ifndef BUILD_LIBTORCH
        torch_npu::profiler::reportMemoryDataToNpuProfiler({ static_cast<int8_t>(c10::DeviceType::PrivateUse1),
            block->device, static_cast<uint8_t>(torch_npu::profiler::MemoryComponentType::CACHING_ALLOCATOR),
            static_cast<uint8_t>(torch_npu::profiler::MemoryDataType::MEMORY_BLOCK_FREE), allocator_type,
            reinterpret_cast<int64_t>(orig_block_ptr), -original_block_size,
            stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            block->stream });
#endif
    }

    /* * combine previously split blocks. returns the size of the subsumed block, or 0 on failure. * */
    size_t try_merge_blocks(Block *dst, Block *src, BlockPool &pool)
    {
        if (!src || src->allocated || src->event_count > 0 || !src->stream_uses.empty() || dst->mapped != src->mapped) {
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
        auto erased = src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
        delete src;
        src = nullptr;

        return subsumed_size;
    }

    BlockPool &get_pool(size_t size, aclrtStream stream)
    {
        // captures_underway is a conservative guess that the current stream may be
        // capturing. It's only non-empty if some thread has begun and not yet ended
        // a capture, so it's usually 0, and we can short-circuit
        // npuStreamCaptureStatus (which does a TLS lookup).
        if (C10_UNLIKELY(!captures_underway.empty())) {
            for (auto &entry : captures_underway) {
                if (entry.second(stream)) {
                    auto it1 = graph_pools.find(entry.first);
                    TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
                    if (size <= kSmallSize) {
                        return it1->second->small_blocks;
                    } else {
                        return it1->second->large_blocks;
                    }
                }
            }
        }
        if (size <= kSmallSize) {
            return small_blocks;
        } else {
            return large_blocks;
        }
    }

    StatTypes get_stat_types_for_pool(const BlockPool &pool)
    {
        StatTypes stat_types = { false };
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
        return stat_types;
    }

    bool should_split(const Block *block, size_t size)
    {
        size_t remaining = block->size - size;
        if (block->pool->is_small || CachingAllocatorConfig::expandable_segments()) {
            return remaining >= kMinBlockSize;
        } else {
            return (size < CachingAllocatorConfig::max_split_size()) && (remaining > kSmallSize);
        }
    }

    static size_t get_allocation_size(size_t size)
    {
        if (size <= kSmallSize) {
            return kSmallBuffer;
        } else if (size < kMinLargeAlloc) {
            return kLargeBuffer;
        } else {
            return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
        }
    }

    bool get_free_block(AllocParams &p)
    {
        BlockPool &pool = *p.pool;

        if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
            // Track block reuse interval only when garbage collection is enabled.
            for (auto &b : pool.blocks) {
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
                auto expandable_size = [](Block *b) {
                    return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
                };
                auto next = it;
                next++;
                while ((*it)->expandable_segment_ && next != pool.blocks.end() && (*next)->stream == p.stream() &&
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
                } while (it != pool.blocks.end() && (*it)->expandable_segment_ && (*it)->stream == p.stream());
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

    bool trigger_free_memory_callbacks(AllocParams &p)
    {
        bool freed_memory = false;
        for (const auto &name : FreeNPUMemoryCallbacksRegistry()->Keys()) {
            if (FreeNPUMemoryCallbacksRegistry()->Create(name) != nullptr) {
                freed_memory |= FreeNPUMemoryCallbacksRegistry()->Create(name)->Execute();
            } else {
                TORCH_CHECK(false, "free memory callback get nullptr", PTA_ERROR(ErrCode::PTR));
            }
        }
        return freed_memory;
    }

    void garbage_collect_cached_blocks(const std::shared_ptr<c10::GatheredContext>& ctx,
                                       std::unique_lock<std::recursive_mutex>& lock)
    {
        // Free unused cached blocks to reclaim NPU memory.
        // Unlike release_cached_blocks(), this does not enforce synchronization and
        // therefore should be of less overheads.

        size_t gc_threshold =
            static_cast<size_t>(CachingAllocatorConfig::garbage_collection_threshold() * allowed_memory_maximum);
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
        for (auto &b : large_blocks.blocks) {
            if (!b->is_split()) {
                total_age += b->gc_count;
                ++freeable_block_count;
            }
        }
        // No free-able blocks?
        if (freeable_block_count == 0) {
            return;
        }

        {
            UnlockGuard guard(lock);
            c10_npu::npuSynchronizeDevice(true);
        }

        // Repeat GC until we reach reclaim > target size.
        bool block_freed = true;
        while (gc_reclaimed < target_size && block_freed && freeable_block_count > 0) {
            // Free blocks exceeding this age threshold first.
            double age_threshold = total_age / freeable_block_count;
            // Stop iteration if we can no longer free a block.
            block_freed = false;

            // Free blocks of > avg age. Don't stop upon reaching the target_size,
            // we don't want this GC to be triggered frequently.
            auto it = large_blocks.blocks.begin();
            while (it != large_blocks.blocks.end()) {
                Block *block = *it;
                ++it;
                if (!block->is_split() && block->gc_count >= age_threshold) {
                    block_freed = true;
                    gc_reclaimed += block->size;
                    total_age -= block->gc_count; // Decrement the age
                    freeable_block_count--;       // One less block that can be freed
                    release_block(block, ctx);

                    ASCEND_LOGD("PTA CachingAllocator gc: free = %zu, cached = %lu, allocated = %lu", block->size,
                        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
                }
            }
        }
    }

    bool alloc_block(AllocParams &p, bool isRetry, const std::shared_ptr<c10::GatheredContext> &ctx,
        std::unique_lock<std::recursive_mutex> &lock)
    {
        size_t size = p.alloc_size;
        void *ptr = nullptr;

        if (isRetry) {
            stats.num_alloc_retries += 1;
        }

        if (set_fraction && total_allocated_memory + size > allowed_memory_maximum) {
            p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
            return false;
        } else if (CachingAllocatorConfig::expandable_segments()) {
            p.block = try_allocate_expandable_block(p.device(), p.stream(), p.pool, p.size(), ctx);
            if (p.block) {
                p.err = ACL_ERROR_NONE;
                if (p.pool->owner_PrivatePool) {
                    // The block is for a NPU graph's PrivatePool.
                    p.pool->owner_PrivatePool->npuMalloc_count++;
                }
            } else {
                p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
            }
            return bool(p.block);
        } else {
            auto active_pool = MemPoolContext::getActiveMemPool();
            if (active_pool && active_pool->allocator() && p.pool->owner_PrivatePool) {
                ptr = active_pool->allocator()->raw_alloc(size);
                p.err = ptr ? ACL_ERROR_NONE : ACL_ERROR_RT_MEMORY_ALLOCATION;
            } else {
                auto policy = aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST;
                if (IsMallocPage1GMem(p.pool->is_small)) {
                    policy = aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE1G_ONLY;
                }
                p.err = c10_npu::acl::AclrtMallocAlign32(&ptr, size, policy);
            }
            if (p.err != ACL_ERROR_NONE) {
                return false;
            }
        }

        ASCEND_LOGD("NPUCachingAllocator malloc by AclrtMallocAlign32: size=%zu", size);

        if (p.pool->owner_PrivatePool) {
            // The block is for a NPU graph's PrivatePool.
            p.pool->owner_PrivatePool->npuMalloc_count++;
        }

        total_allocated_memory += size;
        p.block = new Block(p.device(), p.stream(), size, p.pool, (char *)ptr);
        for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
            update_stat(stats.segment[stat_type], 1);
            update_stat(stats.reserved_bytes[stat_type], size);
        });
        if (size >= CachingAllocatorConfig::max_split_size()) {
            update_stat(stats.oversize_segments, 1);
        }
        ASCEND_LOGD("pta_memory acl_malloc: malloc = %zu, ret = %d", size, p.err);

        // p.block came from new, not npuMalloc. It should not be nullptr here.
        TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);

        record_trace(TraceEntry::SEGMENT_ALLOC, int64_t(p.block->ptr), p.block->size, p.stream(), p.device(), ctx);
        p.block->context_when_segment_allocated = ctx;
        return true;
    }

    /* * Free one or more oversize blocks to the system allocator.  But only enough to satisfy the target size * */
    bool release_available_cached_blocks(const AllocParams& p, const std::shared_ptr<c10::GatheredContext>& ctx,
                                         std::unique_lock<std::recursive_mutex>& lock)
    {
        if (CachingAllocatorConfig::max_split_size() == std::numeric_limits<size_t>::max()) {
            return false;
        }
        BlockPool &pool = *p.pool;
        Block key = p.search_key;
        key.size =
            (key.size < CachingAllocatorConfig::max_split_size()) ? CachingAllocatorConfig::max_split_size() : key.size;
        auto it = pool.blocks.lower_bound(&key);

        {
            UnlockGuard guard(lock);
            c10_npu::npuSynchronizeDevice(true);
        }

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

    // npuSynchronizeDevice must be executed before this function can be called
    bool release_cached_blocks(bool check_error, const std::shared_ptr<c10::GatheredContext> &context, bool free_physical)
    {
        // First ensure that all blocks that can't currently be allocated due to
        // outstanding events are returned to the pool.
        synchronize_and_free_events(check_error, context);

        // Free all non-split cached blocks
        release_blocks(large_blocks, context, free_physical);
        release_blocks(small_blocks, context, free_physical);

        for (auto it = graph_pools_freeable.begin(); it != graph_pools_freeable.end();) {
            // See notifyCaptureDestroy for the strategy here.
            TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
            release_blocks(it->second->small_blocks, context, free_physical);
            release_blocks(it->second->large_blocks, context, free_physical);
            if (it->second->npuMalloc_count == 0) {
                auto erase_count = graph_pools.erase(it->first);
                TORCH_INTERNAL_ASSERT(erase_count == 1);
                it = graph_pools_freeable.erase(it);
            } else {
                ++it;
            }
        }

        return true;
    }

    void release_expandable_segment(Block *block)
    {
        TORCH_INTERNAL_ASSERT(block->size == block->expandable_segment_->size(), "block disagrees with segment",
            PTA_ERROR(ErrCode::INTERNAL));
        TORCH_INTERNAL_ASSERT(!block->mapped, PTA_ERROR(ErrCode::INTERNAL));
        auto it = std::find(expandable_segments_.begin(), expandable_segments_.end(), block->expandable_segment_);
        TORCH_INTERNAL_ASSERT(it != expandable_segments_.end(), PTA_ERROR(ErrCode::INTERNAL));
        expandable_segments_.erase(it);
        block->pool->unmapped.erase(block);
        delete block->expandable_segment_;
        block->expandable_segment_ = nullptr;
        delete block;
        block = nullptr;
    }

    void release_block(Block *block, const std::shared_ptr<c10::GatheredContext> &context)
    {
        TORCH_INTERNAL_ASSERT(!block->expandable_segment_, PTA_ERROR(ErrCode::VALUE));
        ASCEND_LOGD("NPUCachingAllocator free by aclrtFree: size=%zu", block->size);

        record_trace(TraceEntry::SEGMENT_FREE, int64_t(block->ptr), block->size, block->stream, block->device,
            context ? context : block->context_when_segment_allocated);

        auto it = ipc_handle_map.find(block->ptr);
        if (it != ipc_handle_map.end()) {
            NPU_CHECK_ERROR(c10_npu::acl::AclrtIpcMemClose(it->second.data));
            ipc_handle_map.erase(it);
        }

        aclrtFree((void *)block->ptr);
        total_allocated_memory -= block->size;

        auto *pool = block->pool;
        if (pool->owner_PrivatePool) {
            // The npuFreed block belonged to a NPU graph's PrivatePool.
            TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->npuMalloc_count > 0);
            pool->owner_PrivatePool->npuMalloc_count--;
        }

        StatTypes stat_types = get_stat_types_for_pool(*pool);
        for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
            update_stat(stats.segment[stat_type], -1);
            update_stat(stats.reserved_bytes[stat_type], -block->size);
        });

        if (block->size >= CachingAllocatorConfig::max_split_size()) {
            update_stat(stats.oversize_segments, -1);
        }
        ASCEND_LOGD("pta_memory acl_free: free_size = %zu", block->size);

        pool->blocks.erase(block);
        delete block;
        block = nullptr;
    }

    void unmap_block(Block *block, const std::shared_ptr<c10::GatheredContext> &context, bool free_physical)
    {
        auto pool = free_physical ? nullptr : block->pool;
        auto unmapped = block->expandable_segment_->unmap(SegmentRange{ block->ptr, block->size }, pool);
        if (unmapped.size == 0) {
            return;
        }
        block->pool->blocks.erase(block);

        ptrdiff_t before_size = static_cast<char *>(unmapped.ptr) - static_cast<char *>(block->ptr);
        if (before_size > 0) {
            // prev? -> before_free -> block
            Block *before_free = new Block(block->device, block->stream, before_size, block->pool, block->ptr);
            before_free->expandable_segment_ = block->expandable_segment_;
            before_free->splice(block->prev, block);
            block->pool->blocks.insert(before_free);
        }

        auto after_size = block->size - (before_size + unmapped.size);
        if (after_size > 0) {
            // block -> after_free -> next?
            Block *after_free = new Block(block->device, block->stream, after_size, block->pool,
                static_cast<char *>(unmapped.ptr) + unmapped.size);
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
        for_each_selected_stat_type(stat_types,
            [&](size_t stat_type) { update_stat(stats.reserved_bytes[stat_type], -unmapped.size); });

        if (block->pool->owner_PrivatePool) {
            // The npuFreed block belonged to a NPU graph's PrivatePool.
            TORCH_INTERNAL_ASSERT(block->pool->owner_PrivatePool->npuMalloc_count > 0);
            block->pool->owner_PrivatePool->npuMalloc_count--;
        }

        record_trace(TraceEntry::SEGMENT_UNMAP, int64_t(unmapped.ptr), unmapped.size, block->stream, block->device,
            context ? context : block->context_when_segment_allocated);
    }

    void release_blocks(BlockPool &pool, const std::shared_ptr<c10::GatheredContext> &context, bool free_physical)
    {
        std::vector<Block *> to_unmap;
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
        for (Block *block : to_unmap) {
            unmap_block(block, context, free_physical);
            if (!block->prev && !block->next) {
                release_expandable_segment(block);
            }
        }
        // free cached physical handles
        if (free_physical) {
            while (!pool.free_physical_handles_.empty()) {
                aclrtDrvMemHandle handle = pool.free_physical_handles_.back();
                NPU_CHECK_ERROR(c10_npu::acl::AclrtFreePhysical(handle));
                pool.free_physical_handles_.pop_back();
            }
        }
    }

    EventPool::Event create_event_internal(int idx)
    {
        // Leak the event pool to avoid shutdown issues.
        static auto *event_pool = new EventPool();
        return event_pool->get(idx);
    }

    void synchronize_and_free_events(bool check_error, const std::shared_ptr<c10::GatheredContext> &context)
    {
        // This function syncs, so capture should not be underway. Might as well
        // make sure capture-deferred end of life events get processed too.
        TORCH_INTERNAL_ASSERT(captures_underway.empty());
        insert_events_deferred_until_no_capture(context);

        // Synchronize on outstanding events and then free associated blocks.
        for (auto &st : npu_events) {
            for (auto &e : st.second) {
                EventPool::Event event = std::move(e.first);
                Block *block = e.second;
                auto err = aclrtSynchronizeEvent(*event);
                if (err != ACL_ERROR_NONE) {
                    if (check_error) {
                        NPU_CHECK_ERROR(err);
                    } else {
                        ASCEND_LOGE("Event: aclrtSynchronizeEvent failed, event = %p", event.get());
                    }
                } else {
                    ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed, event=%p", event.get());
                }
#ifndef BUILD_LIBTORCH
                const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
                if (C10_UNLIKELY(trigger)) {
                    trigger->traceNpuEventSynchronization(reinterpret_cast<uintptr_t>(event.get()));
                }
#endif

                block->event_count--;
                if (block->event_count == 0) {
                    free_block(block, context);
                }
            }
        }

        npu_events.clear();
    }

    void remove_npugraph_stream_uses(Block *block)
    {
        // remove stream uses added during npugraph capture
        // (i.e., block->stream_uses - block->npugraph_stream_uses)
        if (C10_UNLIKELY(block_to_npugraph_stream_uses.find(block) != block_to_npugraph_stream_uses.end())) {
            stream_set streams(std::move(block->stream_uses));
            AT_ASSERT(block->stream_uses.empty());
            for (auto &stream : streams) {
                if (block_to_npugraph_stream_uses[block].find(stream) == block_to_npugraph_stream_uses[block].end()) {
                    block->stream_uses.insert(stream);
                }
            }
            block_to_npugraph_stream_uses.erase(block);
        }
    }

    void insert_events(Block *block)
    {
        int pre_device = -1;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&pre_device));
        aclrtContext compiler_ctx = aclrtContext();
        aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);
        NPU_CHECK_ERROR(aclrtSetCurrentContext(c10_npu::GetDeviceContext(block->device)));

        stream_set streams(std::move(block->stream_uses));
        AT_ASSERT(block->stream_uses.empty(), PTA_ERROR(ErrCode::VALUE));
        for (auto &stream : streams) {
            NPU_CHECK_ERROR(c10_npu::SetDevice(stream.device_index()));

            EventPool::Event event = create_event_internal(stream.device_index());
            event->record(stream);
            ASCEND_LOGI("Event: record DeviceAllocator is successfully executed, event=%p", event.get());

            block->event_count++;
            npu_events[stream].emplace_back(std::move(event), block);
        }
        if (ret_ctx == ACL_ERROR_NONE) {
            NPU_CHECK_ERROR(aclrtSetCurrentContext(compiler_ctx)); 
            // Setting context will exchange device implicitly, so we need to reset the cached device here to ensure consistency.
            NPU_CHECK_ERROR(c10_npu::SetDevice(pre_device));
        }
    }

    void insert_events_deferred_until_no_capture(const std::shared_ptr<c10::GatheredContext> &context)
    {
        if (C10_UNLIKELY(!needs_events_deferred_until_no_capture.empty())) {
            for (auto *block : needs_events_deferred_until_no_capture) {
                TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
                // only streams recorded before npugraph will be used to insert events
                // since we know all streams recorded during npugraph must have
                // completed (refer to Section 3.2.8.7.3.1 Cross-stream Dependencies and
                // Events in CUDA Programming Guide).
                remove_npugraph_stream_uses(block);
                insert_events(block);
                if (block->event_count == 0) {
                    free_block(block, context);
                }
            }
            needs_events_deferred_until_no_capture.clear();
        }
    }

    void process_events(const std::shared_ptr<c10::GatheredContext> &context)
    {
        insert_events_deferred_until_no_capture(context);

        // Process outstanding npuEvents. Events that are completed are removed
        // from the queue, and the 'event_count' for the corresponding allocation
        // is decremented. Stops at the first event which has not been completed.
        // Since events on different devices or streams may occur out of order,
        // the processing of some events may be delayed.
        for (auto it = npu_events.begin(); it != npu_events.end();) {
            while (!it->second.empty()) {
                auto &e = it->second.front();
                EventPool::Event event = std::move(e.first);
                Block *block = e.second;

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
    void cache_info_aux(BlockPool &blocks, size_t *total, size_t *largest)
    {
        for (auto it = blocks.blocks.begin(); it != blocks.blocks.end(); ++it) {
            size_t blocksize = (*it)->size;
            *total += blocksize;
            if (blocksize > *largest) {
                *largest = blocksize;
            }
        }
    }

    void record_trace(TraceEntry::Action action, int64_t addr, size_t size, aclrtStream stream, int device,
        std::shared_ptr<c10::GatheredContext> context)
    {
        if (!record_history) {
            return;
        }

        auto te = TraceEntry(action, device, addr, size, stream,
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

static void uncached_delete(void *ptr)
{
    if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        c10_npu::npuSynchronizeDevice(false);
    }
    ASCEND_LOGD("Without NPUCachingAllocator, free by aclrtFree.");
    NPU_CHECK_ERROR(aclrtFree(ptr));
}

void local_raw_delete(void *ptr);

class NpuCachingAllocator : public NPUAllocator {
private:
    std::mutex mutex;

    // allocated blocks by device pointer
    ska::flat_hash_map<void *, Block *> allocated_blocks;

    void add_allocated_block(Block *block)
    {
        std::lock_guard<std::mutex> lock(mutex);
        allocated_blocks[block->ptr] = block;
    }

public:
    std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

    Block *get_allocated_block(void *ptr, bool remove = false)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end()) {
            return nullptr;
        }
        Block *block = it->second;
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
    /* * allocates a block which is safe to use from the provided stream */
    void malloc(void **devPtr, int device, size_t size, aclrtStream stream)
    {
        TORCH_INTERNAL_ASSERT(0 <= device && static_cast<size_t>(device) < device_allocator.size(),
            "Allocator not initialized for device ", device, ": did you call init?", PTA_ERROR(ErrCode::PARAM));
        Block *block = device_allocator[device]->malloc(device, size, stream);

        add_allocated_block(block);
        *devPtr = static_cast<void *>(block->ptr);
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuMemoryAllocation(reinterpret_cast<uintptr_t>(*devPtr));
        }
#endif
    }

    void free(void *ptr)
    {
        if (!ptr) {
            return;
        }
        Block *block = get_allocated_block(ptr, true);
        if (!block) {
            AT_ERROR("invalid device pointer: ", ptr);
        }
#ifndef BUILD_LIBTORCH
        const c10_npu::impl::PyCallbackTrigger *trigger = c10_npu::impl::NPUTrace::getTrace();
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuMemoryDeallocation(reinterpret_cast<uintptr_t>(block->ptr));
        }
#endif
        auto orig_block_ptr = block->ptr;
        auto orig_block_size = block->size;
        device_allocator[block->device]->free(block);
    }

    void setMemoryFraction(double fraction, int device) override
    {
        TORCH_INTERNAL_ASSERT(0 <= device && device < device_allocator.size(), "Allocator not initialized for device ",
            device, ": did you call init?", PTA_ERROR(ErrCode::PARAM));
        TORCH_INTERNAL_ASSERT(0 <= fraction && fraction <= 1, "invalid fraction:", fraction,
            ". Please set within (0, 1).", PTA_ERROR(ErrCode::PARAM));

        c10_npu::SetDevice(device);

        device_allocator[device]->setMemoryFraction(fraction);
    }

    void recordHistory(bool enabled, CreateContextFn context_recorder, size_t alloc_trace_max_entries,
        RecordContext when) override
    {
        for (auto &allocator : device_allocator) {
            allocator->recordHistory(enabled, context_recorder, alloc_trace_max_entries, when);
        }
    }

    bool isHistoryEnabled() override
    {
        int device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        return device_allocator[device]->isHistoryEnabled();
    }

  bool checkPoolLiveAllocations(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) override
  {
      return device_allocator[device]->checkPoolLiveAllocations(mempool_id, expected_live_allocations);
  }

    void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override
    {
        for (auto &allocator : device_allocator) {
            allocator->attachOutOfMemoryObserver(observer);
        }
    }

    bool checkUceInMemPool(int device) override
    {
        return device_allocator[device]->checkUceInMemPool();
    }

    bool checkBlockIsSafe(const c10::DataPtr &ptr) override
    {
        if (!ptr.get()) {
            return true;
        }
        if (ptr.get_deleter() != &local_raw_delete) {
            return true;
        }
        Block *block = get_allocated_block(ptr.get());
        TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found", PTA_ERROR(ErrCode::NOT_FOUND));
        return block->is_safe;
    }

    void markAllBlockUnsafe(int device) override
    {
        return device_allocator[device]->markAllBlockUnsafe();
    }

    void updateBlockToSafe(const c10::DataPtr &ptr) override
    {
        if (!ptr.get()) {
            return;
        }
        if (ptr.get_deleter() != &local_raw_delete) {
            return;
        }
        Block *block = get_allocated_block(ptr.get());
        TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found", PTA_ERROR(ErrCode::NOT_FOUND));
        if (block->is_safe == false) {
            ASCEND_LOGI("Triggers to refresh the data of the unsafe memory block and remove the unsafe flag");
        }
        block->is_safe = true;
    }

    void cleanEvent() override
    {
        int count = static_cast<int>(device_allocator.size());
        for (int i = 0; i < count; i++) {
            device_allocator[i]->release_and_free_events();
        }
    }

    void emptyCacheImpl(bool check_error, bool free_physical) override
    {
        ASCEND_LOGD("Begin empty cache with check_error = %d", check_error);
        int32_t current_device = 0;
        if (check_error) {
            NPU_CHECK_ERROR(c10_npu::GetDevice(&current_device));
        } else {
            NPU_CHECK_WARN(c10_npu::GetDevice(&current_device));
        }
        auto used_devices_list = c10_npu::GetUsedDevices();
        for (int8_t device_idx : used_devices_list) {
            if (check_error) {
                NPU_CHECK_ERROR(c10_npu::SetDevice(device_idx));
            } else {
                NPU_CHECK_WARN(c10_npu::SetDevice(device_idx));
            }
            device_allocator[device_idx]->emptyCache(device_idx, check_error, free_physical);
        }
        if (check_error) {
            NPU_CHECK_ERROR(c10_npu::MaybeSetDevice(current_device));
        } else {
            NPU_CHECK_WARN(c10_npu::MaybeSetDevice(current_device));
        }
        ASCEND_LOGD("End empty cache with check_error = %d", check_error);
    }

    void emptyCache(bool check_error) override
    {
        emptyCacheImpl(check_error, true);
    }

    void emptyVirtAddrCache(bool check_error) override
    {
        if (!CachingAllocatorConfig::expandable_segments()) {
            AT_ERROR("Unsupported config for empty_virt_addr_cache, please enable expandable_segments.");
        }
        emptyCacheImpl(check_error, false);
    }

    void *getBaseAllocation(void *ptr, size_t *outSize) override
    {
        Block *block = get_allocated_block(ptr);
        if (!block) {
            AT_ERROR("invalid device pointer: ", ptr);
        }
        return device_allocator[block->device]->getBaseAllocation(block, outSize);
    }

    ShareableHandle shareIpcHandle(void* ptr) override
    {
        Block* block = get_allocated_block(ptr);
        if (!block) {
            AT_ERROR("invalid device pointer: ", ptr);
        }
        return device_allocator[block->device]->shareIpcHandle(block);
    }

    void recordStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream) override
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

        Block *block = get_allocated_block(ptr.get());
        // block must not be null reaching here
        TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found", PTA_ERROR(ErrCode::NOT_FOUND));
        device_allocator[block->device]->recordStream(block, stream);
    }

    void eraseStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream)
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

        Block *block = get_allocated_block(ptr.get());
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

    void eraseStreamWithBlockPtr(void* block_ptr, c10_npu::NPUStream stream, void* work_ptr) override
    {
        Block* block = static_cast<Block*>(block_ptr);
        if (!block) {
            AT_ERROR("invalid block pointer");
        }

        if (!work_ptr) {
            AT_ERROR("invalid hccl work pointer");
        }

        if (block->stream != c10_npu::getCurrentNPUStream(block->device).stream(false) || block->hccl_work_ptr != work_ptr) {
            // If the Stream applying for tensor block different from
            // the stream of submiting event wait task in HCCL synchronize()
            // method, the recordSteam can not be erased.
            // New tensor creation may use the block before HCCL op is complete.
            return;
        }

        device_allocator[block->device]->eraseStream(block, stream);
    }

    void* getBlockPtr(const c10::DataPtr& ptr) override
    {
        if (!ptr.get()) {
            return nullptr;
        }
        Block* block = get_allocated_block(ptr.get());
        return static_cast<void*>(block);
    }

    void recordHcclWorkForBlock(void* block_ptr, void* work_ptr) override
    {
        Block* block = static_cast<Block*>(block_ptr);
        block->hccl_work_ptr = work_ptr;
        return;
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

    std::shared_ptr<AllocatorState> getCheckpointState(
        c10::DeviceIndex device,
        MempoolId_t id) override
    {
        return device_allocator[device]->getCheckpointState(id);
    }

    /**
     * @brief Checkpoint the private pool state identified in `as` to its prior
     * state
     *
     * @param device - device of the pool to manipulate
     * @param as - allocator state
     * @param stale_live_storages - storages of tensors which are currently
     * allocated but which will be not be allocated after the checkpoint is set.
     * For these storages we will remove their deleter function.
     * @return CheckpointDelta - Freed Pointers and DataPtrs that contain deleter
     * functions for all allocated blocks in the new checkpoint state.
     */
    CheckpointDelta setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<AllocatorState> as) override
    {
        std::shared_ptr<PrivatePoolState> pps = std::dynamic_pointer_cast<PrivatePoolState>(as);

        TORCH_CHECK(pps, "Expected PrivatePoolState", PTA_ERROR(ErrCode::PARAM));

        auto rr = device_allocator[device]->setCheckpointPoolState(*pps);

        CheckpointDelta cpd;
        for (void* ptr : rr.allocations_freed) {
            // remove block
            get_allocated_block(ptr, true);
            cpd.ptrs_freed.push_back(ptr);
        }
        for (Block* block : rr.allocations_created) {
            add_allocated_block(block);
            cpd.dataptrs_allocd.emplace_back(
                block->ptr,
                block->ptr,
                &local_raw_delete,
                c10::Device(c10::DeviceType::PrivateUse1, device));
        }

        return cpd;
    }

    void beginAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id,
        std::function<bool(aclrtStream)> filter) override
    {
        assertValidDevice(device);
        device_allocator[device]->beginAllocateToPool(std::move(mempool_id), std::move(filter));
    }

    void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) override
    {
        assertValidDevice(device);
        device_allocator[device]->endAllocateToPool(mempool_id);
    }

    void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override
    {
        assertValidDevice(device);
        device_allocator[device]->releasePool(std::move(mempool_id));
    }

    c10::DataPtr allocate(size_t size) override
    {
        constexpr size_t one_exa_bytes = 1152921504606846976ULL;
        if (size >= one_exa_bytes) {
            auto retmsg = std::string("NPU out of memory. Tried to allocate more than 1EB memory.");
            ASCEND_LOGE("%s", retmsg.c_str());
            TORCH_CHECK_WITH(OutOfMemoryError, false, retmsg.c_str());
        }
        
        int device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        LazySetDevice(device);
        void *devPtr = nullptr;
        void (*deleteFunc)(void *) = &local_raw_delete;

        if (size != 0) {
            if (c10_npu::option::OptionsManager::CheckForceUncached()) {
                deleteFunc = &uncached_delete;
                size_t alloc_size = size + AddPadSize();
                NPU_CHECK_ERROR(c10_npu::acl::AclrtMallocAlign32(&devPtr, alloc_size,
                    aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
                ASCEND_LOGD("Without NPUCachingAllocator, malloc by "
                    "AclrtMallocAlign32: size=%zu",
                    alloc_size);
            } else {
                this->malloc(&devPtr, device, size, c10_npu::getCurrentNPUStreamNoWait(device));
            }
        }
        return { devPtr, devPtr, deleteFunc, c10::Device(c10::DeviceType::PrivateUse1, device) };
    }

    c10::DataPtr allocate_with_aligned(size_t size, size_t base_addr_aligned_kb) const override
    {
        constexpr size_t one_exa_bytes = 1152921504606846976ULL;
        if (C10_UNLIKELY(size >= one_exa_bytes)) {
            auto retmsg = std::string("NPU out of memory. Tried to allocate more than 1EB memory.");
            ASCEND_LOGE("%s", retmsg.c_str());
            TORCH_CHECK_WITH(OutOfMemoryError, false, retmsg.c_str());
        }
        int device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        void *realPtr = nullptr;
        void (*deleteFunc)(void *) = &local_raw_delete;

        size_t aligned = base_addr_aligned_kb * 1024;
        if (size != 0) {
            if (c10_npu::option::OptionsManager::CheckForceUncached()) {
                deleteFunc = &uncached_delete;
                size_t alloc_size = size + AddPadSize() + aligned;
                NPU_CHECK_ERROR(c10_npu::acl::AclrtMallocAlign32(&realPtr, alloc_size,
                                                                 aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
                ASCEND_LOGD("Without NPUCachingAllocator, malloc by "
                            "AclrtMallocAlign32: size=%zu", alloc_size);
            } else {
                const_cast<NpuCachingAllocator *>(this)->malloc(&realPtr, device, size + aligned,
                                                                c10_npu::getCurrentNPUStreamNoWait(device));
            }
        }
        void *devPtr = reinterpret_cast<void*>(aligned * ((reinterpret_cast<uintptr_t>(realPtr) + aligned - 1) / aligned));
        return { devPtr, realPtr, deleteFunc, c10::Device(c10::DeviceType::PrivateUse1, device) };
    }

    c10::DeleterFnPtr raw_deleter() const override
    {
        if (c10_npu::option::OptionsManager::CheckForceUncached()) {
            return &uncached_delete;
        } else {
            return &local_raw_delete;
        }
    }

    void cacheInfo(int dev_id, size_t *cachedAndFree, size_t *largestBlock) override
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

    void *raw_alloc(size_t nbytes) override
    {
        if (nbytes == 0) {
            return nullptr;
        }
        int device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        void *r = nullptr;
        malloc(&r, device, nbytes, c10_npu::getCurrentNPUStreamNoWait(device));
        return r;
    }

    void *raw_alloc_with_stream(size_t nbytes, aclrtStream stream) override
    {
        if (nbytes == 0) {
            return nullptr;
        }
        int device;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
        void *r = nullptr;
        malloc(&r, device, nbytes, stream);
        return r;
    }

    void raw_delete(void *ptr) override
    {
        this->free(ptr);
    }

    std::mutex IpcMutex;
    struct MemHandleCacheEntry {
        MemHandleCacheEntry(
            c10::DeviceIndex device,
            std::string& handle,
            const DeviceCachingAllocator& allocator)
            : device_(device)
        {
            int type = SHAREABLE_NPU_MALLOC;
            std::istringstream ss(handle);
            if (handle.size() != ACL_IPC_HANDLE_SIZE) {
                auto version = ss.get();
                TORCH_CHECK(
                    version <= SHAREABLE_HANDLE_VERSION,
                    "received sharable handle from a future version of torch that this version does not know how to handle",
                    PTA_ERROR(ErrCode::NOT_SUPPORT));
                type = ss.get();
            }
            // otherwise this is coming from an old pytorch where it has to be a raw
            // SHAREABLE_NPU_MALLOC
            if (type == SHAREABLE_NPU_MALLOC) {
                handle_str handle_r;
                ss.read(handle_r.data, ACL_IPC_HANDLE_SIZE);
                NPU_CHECK_ERROR(c10_npu::acl::AclrtIpcMemImportByKey(
                    &npu_ipc_ptr_, handle_r.data, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS));
                handle_s.assign(handle_r.data, ACL_IPC_HANDLE_SIZE);
            } else if (type == SHAREABLE_NPU_EXPANDABLE_SEGMENT) {
                expandable_segment_ =
                    ExpandableSegment::fromShared(device, ss)
                        .release();
            } else {
                TORCH_INTERNAL_ASSERT(
                    false, "Unexpected or illformed shareable handle type");
            }
        }
        // this struct expects that clear is explicitly called to
        // free resources, because we only want this code running when
        // the shared pointer to this entry is destructed, not during
        // deinitialization when npu may already have been shutdown.
        // This replicates the previous behavior of this map when it
        // stored raw npu_ipc_ptr_ handles.
        void clear()
        {
            if (npu_ipc_ptr_) {
                {
                    c10_npu::NPUGuard device_guard(device_);
                    c10_npu::npuSynchronizeDevice(true);
                }
                NPU_CHECK_ERROR(c10_npu::acl::AclrtIpcMemClose(handle_s.c_str()));
                npu_ipc_ptr_ = nullptr;
            }
            if (expandable_segment_) {
                delete expandable_segment_;
                expandable_segment_ = nullptr;
            }
        }
        void* ptr()
        {
            if (npu_ipc_ptr_) {
                return npu_ipc_ptr_;
            }
            if (expandable_segment_) {
                return expandable_segment_->ptr();
            }
            return nullptr;
        }
        c10::DeviceIndex device_;
        ExpandableSegment* expandable_segment_{nullptr};
        void* npu_ipc_ptr_{nullptr}; // nullptr if expandable_segment_ is not null
        std::weak_ptr<void> wp_;
        std::string handle_s;
    };
    ska::flat_hash_map<std::string, MemHandleCacheEntry> ipcMemHandle_to_devptr;

    std::shared_ptr<void> getIpcDevPtr(std::string handle) override
    {
        std::lock_guard<std::mutex> lock(IpcMutex);

        auto iter = ipcMemHandle_to_devptr.find(handle);
        if (iter != ipcMemHandle_to_devptr.end()) {
            auto devptr = iter->second.wp_.lock();
            TORCH_INTERNAL_ASSERT(devptr, "entry in cache has missing shared_ptr");
            return devptr;
        }
        int curr_device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&curr_device));
        auto inserted = ipcMemHandle_to_devptr.insert(
            iter,
            {handle,
            MemHandleCacheEntry(
                static_cast<c10::DeviceIndex>(curr_device), handle, *device_allocator[curr_device])});
        auto sp = std::shared_ptr<void>(
            inserted->second.ptr(), [handle, this](void* ptr) {
                std::unique_lock<std::mutex> deleter_lock(IpcMutex);

                auto it = ipcMemHandle_to_devptr.find(handle);
                TORCH_INTERNAL_ASSERT(it != ipcMemHandle_to_devptr.end());
                auto entry = std::move(it->second);
                ipcMemHandle_to_devptr.erase(it);

                // ExpandableSegment synchronizes on destruction in unmapHandles, so
                // we need to release the lock first to minimize the performance hit.
                deleter_lock.unlock();
                entry.clear();
            });
        inserted->second.wp_ = sp;
        return sp;
    }

    void FreeDeviceCachedMemory(int device) override
    {
        device_allocator[device]->emptyCache(device, true, true);
    }

    std::string name() override
    {
        return "native";
    }

    // Note [COW/lazy_clone is not supported yet]
    void copy_data(void *dest, const void *src, std::size_t count) const final
    {
        default_copy_data(dest, src, count);
    }

    void buildServerMemMapForHccl(int device, std::shared_ptr<c10d_npu::HCCLComm> hcclComm)
    {
        device_allocator[device]->buildServerMemMapForHccl(hcclComm);
    }
};

NpuCachingAllocator caching_allocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &caching_allocator);


void local_raw_delete(void *ptr)
{
    caching_allocator.free(ptr);
}

void *MallocBlock(size_t size, void *stream, int device)
{
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
    return reinterpret_cast<void *>(block);
}

void FreeBlock(void *handle)
{
    Block *block = reinterpret_cast<Block *>(handle);
    AT_ASSERT(block, PTA_ERROR(ErrCode::PTR));
    caching_allocator.assertValidDevice(block->device);
    AT_ASSERT(caching_allocator.device_allocator[block->device], PTA_ERROR(ErrCode::NOT_FOUND));
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;
    caching_allocator.device_allocator[block->device]->free(block,
        static_cast<uint8_t>(torch_npu::profiler::MemoryAllocatorType::ALLOCATOR_EXTERNAL));
}

void *GetBlockPtr(const void *handle)
{
    const Block *block = reinterpret_cast<const Block *>(handle);
    AT_ASSERT(block, PTA_ERROR(ErrCode::PTR));
    return block->ptr;
}

size_t GetBlockSize(const void *handle)
{
    const Block *block = reinterpret_cast<const Block *>(handle);
    AT_ASSERT(block, PTA_ERROR(ErrCode::PTR));
    return block->size;
}

struct BackendStaticInitializer {
    BackendStaticInitializer()
    {
        allocator.store(&caching_allocator);
    }
};

std::atomic<NPUAllocator *> allocator;
BackendStaticInitializer backend_static_initializer;

std::mutex *getFreeMutex()
{
    static std::mutex npu_free_mutex;
    return &npu_free_mutex;
}
} // namespace NPUCachingAllocator
} // namespace c10_npu

namespace c10_npu {
// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10_npu::MemPool().
//
// uuid_ is incremented when NPUGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or NPUGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> MemPool::uid_{ 1 };
std::atomic<CaptureId_t> MemPool::uuid_{ 1 };


MemPool::MemPool(NPUCachingAllocator::NPUAllocator *allocator, bool is_user_created)
    : allocator_(allocator), is_user_created_(is_user_created)
{
    if (is_user_created_) {
        id_ = { 0, uid_++ };
    } else {
        id_ = { uuid_++, 0 };
    }
}

MempoolId_t MemPool::id()
{
    return id_;
}

NPUCachingAllocator::NPUAllocator *MemPool::allocator()
{
    return allocator_;
}

// Note that active_mempool_ is a global variable here
// and not inside MemPoolContext class, because in windows we
// can't use __declspec(dllexport) and __declspec(thread)
static thread_local MemPool *active_mempool_ = nullptr;

MemPoolContext::MemPoolContext(MemPool *mempool) : prev_mempool_(active_mempool_)
{
    active_mempool_ = mempool;
}

MemPoolContext::~MemPoolContext()
{
    active_mempool_ = prev_mempool_;
}

MemPool *MemPoolContext::getActiveMemPool()
{
    return active_mempool_;
}
} // namespace c10_npu
