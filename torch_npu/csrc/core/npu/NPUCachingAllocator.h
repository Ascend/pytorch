#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"


#include <mutex>

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


void* raw_alloc(size_t nbytes);
void* raw_alloc_with_stream(size_t nbytes, aclrtStream stream);
void raw_delete(void* ptr);

c10::Allocator* get();
void init();
void setMemoryFraction(double fraction, int device);
C10_NPU_API void emptyCache(bool check_error = true);
C10_NPU_API void setShutdownStats();
void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
void* getBaseAllocation(void* ptr, size_t* size);
void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream);
void eraseStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream);
DeviceStats getDeviceStats(int device);
void resetAccumulatedStats(int device);
void resetPeakStats(int device);
std::vector<SegmentInfo> snapshot();

std::mutex* getFreeMutex();

void FreeDeviceCachedMemory(int device);

std::string name();
} // namespace NPUCachingAllocator
} // namespace c10_npu
