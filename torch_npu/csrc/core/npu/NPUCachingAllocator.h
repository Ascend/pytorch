#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"


#include <mutex>
#include <atomic>

namespace c10_npu {
namespace NPUCachingAllocator {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class FreeMemoryCallback {
public:
  virtual ~FreeMemoryCallback(){};
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeNPUMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeNPUMemoryCallbacksRegistry, name, __VA_ARGS__);


// (Ascend): Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (NPUCachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THNCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.
struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3  // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;
// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from npuMalloc().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be released via npuFree)
  StatArray inactive_split;

  // SUM: bytes requested by client code
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to NPU malloc necessitating cache flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to NPU after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

// Struct containing info of an allocation block (i.e. a fractional part of a cudaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  int64_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  uintptr_t  address = 0;
  int64_t total_size = 0;
  int64_t requested_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  bool is_expandable = false;
  std::vector<BlockInfo> blocks;
};


class NPUAllocator : public c10::Allocator {
public:
    virtual std::mutex* getFreeMutex() const = 0;
    virtual void* raw_alloc(size_t nbytes) = 0;
    virtual void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream) = 0;
    virtual void raw_delete(void* ptr) = 0;
    virtual void init(int device_count) = 0;
    virtual bool initialized() = 0;
    virtual void setMemoryFraction(double fraction, int device) = 0;
    virtual void emptyCache(bool check_error) = 0;
    virtual void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) = 0;
    virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
    virtual void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) = 0;
    virtual void eraseStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) = 0;
    virtual DeviceStats getDeviceStats(int device) = 0;
    virtual void resetAccumulatedStats(int device) = 0;
    virtual void resetPeakStats(int device) = 0;
    virtual std::vector<SegmentInfo> snapshot() = 0;
    virtual void FreeDeviceCachedMemory(int device) = 0;
    virtual void setShutdownStats() = 0;
    virtual std::string name() = 0;
};

// Allocator object, statically initialized
// See BackendInitializer in CUDACachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
C10_NPU_API extern std::atomic<NPUAllocator*> allocator;


inline NPUAllocator* get()
{
    return allocator.load();
}

// Called directly by clients.
inline std::mutex* getFreeMutex()
{
    return get()->getFreeMutex();
}

inline void* raw_alloc(size_t nbytes)
{
    return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream)
{
    return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr)
{
    return get()->raw_delete(ptr);
}

inline void init()
{
    uint32_t device_count = 0;
    NPU_CHECK_ERROR(aclrtGetDeviceCount(&device_count));
    return get()->init(device_count);
}

inline void setMemoryFraction(double fraction, int device)
{
    return get()->setMemoryFraction(fraction, device);
}

C10_NPU_API inline void emptyCache(bool check_error = true)
{
    return get()->emptyCache(check_error);
}

inline void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock)
{
    return get()->cacheInfo(dev_id, cachedAndFree, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size)
{
    return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream)
{
    return get()->recordStream(ptr, stream);
}

inline void eraseStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream)
{
    return get()->eraseStream(ptr, stream);
}

inline DeviceStats getDeviceStats(int device)
{
    return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(int device)
{
    return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(int device)
{
    return get()->resetPeakStats(device);
}

inline std::vector<SegmentInfo> snapshot()
{
    return get()->snapshot();
}

inline void FreeDeviceCachedMemory(int device)
{
    return get()->FreeDeviceCachedMemory(device);
}

C10_NPU_API inline void setShutdownStats()
{
    return get()->setShutdownStats();
}

inline std::string name()
{
    return get()->name();
}

} // namespace NPUCachingAllocator
} // namespace c10_npu
