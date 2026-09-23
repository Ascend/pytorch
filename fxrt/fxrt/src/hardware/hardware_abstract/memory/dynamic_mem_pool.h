#ifndef FXRT_SRC_HARDWARE_MEMORY_DYNAMIC_MEM_POOL_H_
#define FXRT_SRC_HARDWARE_MEMORY_DYNAMIC_MEM_POOL_H_

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <tuple>

#include "common/visible.h"
#include "hardware/hardware_abstract/memory/mem_pool_util.h"
#include "hardware/hardware_abstract/stream_util.h"
#include "hardware/hardware_abstract/device_event.h"
#include "common/logger.h"

namespace fxrt {
namespace device {
constexpr int kShiftOffset = 2;
// Alloc memory aligned according to 512 bytes.
constexpr size_t kDynamicMemAlignSize = 512;
// The minimum unit size (1G) of memory block used for dynamic extend.
constexpr size_t kDynamicMemAllocUnitSize = 1024 << 20;

const char kPersistentParamMem[] = "Persistent mem";
const char kCommonMem[] = "Common mem";
constexpr size_t kMBToByte = 1024 << 10;
constexpr size_t kGBToByte = 1024 << 20;
// The smallest memory request size, if it is smaller than this size, the device memory request may fail
// Set experience value to 10M
const size_t kMinimumAllocMem = 10 << 20;

const char kBlockMemorySize[] = "block_memory_size";
const char kBlockStreamId[] = "block_stream_id";
const char kCommonMemPoolType[] = "common_mem_pool";
const char kPersistentMemPoolType[] = "persistent_mem_pool";
using MallocFuncType = void*(size_t, int, void*);
using FreeFuncType = void(void*, size_t, int, void*);

// The status of memory buf.
enum class FXRT_EXPORT DynamicMemBufStatus : int { kMemBufIdle, kMemBufUsed, kMemBufEagerFree, kMemBufUsedByEvent };
FXRT_EXPORT const std::string& DynamicMemBufStatusToString(DynamicMemBufStatus status);

using DeviceMemPtr = void(*);
struct DeviceAddrCmp {
  bool operator()(const DeviceMemPtr& addr1, const DeviceMemPtr& addr2) const {
    return addr1 < addr2;
  }
};

// The AllocatorDebugInfo wrapper which is the local thread for the dynamic memory pool.
class FXRT_EXPORT DynamicMemAllocatorDebugInfo;
// Memory buf is the smallest operation object of dynamic memory pool.
struct FXRT_EXPORT DynamicMemBuf;
using DynamicMemBufPtr = std::shared_ptr<DynamicMemBuf>;
// Multimap key is the tensor size, for finding the idle memory buf by tensor size.
using SizeMapMemBuf = std::multimap<size_t, DynamicMemBufPtr>;
// Map key is the device address, for finding the used memory buf in memory block by device address.
using DeviceAddrMapMemBuf = std::map<DeviceMemPtr, DynamicMemBufPtr, DeviceAddrCmp>;
// Memory block is composed of memory buf.
class FXRT_EXPORT DynamicMemBlock;
using DynamicMemBlockPtr = std::shared_ptr<DynamicMemBlock>;

struct FXRT_EXPORT MemStatusManager;
using MemStatusManagerPtr = std::shared_ptr<MemStatusManager>;

// Help class for unordered_map, pair has no hash method, need override it.
struct pair_hash {
  template <class L, class R>
  std::size_t operator()(const std::pair<L, R>& param) const {
    size_t hash = std::hash<L>{}(param.first);
    hash <<= (sizeof(size_t) << kShiftOffset);
    hash ^= std::hash<R>{}(param.second);
    return std::hash<size_t>{}(hash);
  }
};

struct FXRT_EXPORT MemBuf;

// Interface of dynamic memory pool.
class FXRT_EXPORT DynamicMemPool {
 public:
  virtual ~DynamicMemPool() = default;

  // Initialize memory pool with init size, increase size and max size.
  virtual void Initialize(size_t initSize, size_t increaseSize, size_t maxSize) {}

  // Release the real device memory.
  virtual void ReleaseDeviceRes() {
    RT_GLOG(ERROR) << "Not implemented";
  }

  // The main program entry of memory alloc.
  virtual DeviceMemPtr AllocTensorMem(
      size_t size,
      bool fromPersistentMem = false,
      bool needRecycle = false,
      uint32_t streamId = kDefaultStreamIndex) {
    RT_GLOG(ERROR) << "Not implemented";
    return nullptr;
  }

  // The main program entry of continuous memory alloc.
  virtual std::vector<DeviceMemPtr> AllocContinuousTensorMem(
      const std::vector<size_t>& sizeList,
      uint32_t streamId = kDefaultStreamIndex) {
    RT_GLOG(ERROR) << "Not implemented";
    return {};
  }
  // The main program entry of memory free.
  virtual void FreeTensorMem(const DeviceMemPtr& deviceAddr) {
    RT_GLOG(ERROR) << "Not implemented";
  }

  virtual bool DoFreeTensorMem(const DeviceMemPtr& deviceAddr) {
    return false;
  }

  // The main program entry of part memorys free and part memorys keep.
  virtual void FreePartTensorMems(
      const std::vector<DeviceMemPtr>& freeAddrs,
      const std::vector<DeviceMemPtr>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) {
    RT_GLOG(ERROR) << "Not implemented";
  }

  // Help method for dynamic memory proxy.
  virtual std::vector<MemBuf*> DoFreePartTensorMems(
      const std::vector<DeviceMemPtr>& freeAddrs,
      const std::vector<DeviceMemPtr>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) {
    return {};
  }

  virtual size_t EmptyCache() {
    return -1L;
  }

  virtual size_t ReleaseFreeBlocks() {
    return -1L;
  }

  // Element in vector : memoryStreamId, address
  virtual bool RecordEvent(
      int64_t taskIdOnStream,
      uint32_t userStreamId,
      const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
      const DeviceEventPtr& event) {
    return false;
  }

  virtual bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) {
    return false;
  }

  virtual bool WaitEvent(int64_t taskIdOnStream, uint32_t memoryStreamId) {
    return false;
  }

  virtual bool SyncAllEvents() {
    return false;
  }

  // The real size by memory alloc aligned.
  virtual size_t AlignMemorySize(size_t size) const {
    if (size == 0) {
      return kDynamicMemAlignSize;
    }
    return ((size + kDynamicMemAlignSize - 1) / kDynamicMemAlignSize) * kDynamicMemAlignSize;
  }

  // Calculate memory block required alloc size when adding the memory block.
  virtual size_t CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool needRecycle = false) {
    return kDynamicMemAllocUnitSize;
  }

  // Set mem pool block size
  virtual void SetMemPoolBlockSize(size_t availableDeviceMemSize) {}

  // Get the minimum memory unit size using for dynamic extend.
  virtual size_t MemAllocUnitSize(bool fromPersistentMem) const {
    return kDynamicMemAllocUnitSize;
  }

  virtual void SetMemAllocUintSize(size_t commonSize, size_t persistSize = kDynamicMemAllocUnitSize) {}

  virtual void* GetMinUsingMemoryAddr() const {
    return nullptr;
  }

  // The related interface of device memory real operation, needs override by device type.
  virtual size_t AllocDeviceMem(size_t size, DeviceMemPtr* addr) {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual bool FreeDeviceMem(const DeviceMemPtr& addr) {
    RT_GLOG(ERROR) << "Not implemented";
    return false;
  }

  virtual size_t free_mem_size() {
    return 0;
  }

  virtual uint64_t total_mem_size() const {
    return 0;
  }

  virtual size_t GetMaxUsedMemSize() const {
    return 0;
  }

  virtual size_t GetVmmUsedMemSize() const {
    return 0;
  }

  // The related interface of device memory eager free.
  virtual void DefragMemory() {}

  // Display the brief state information of memory block and memory buf.
  virtual void DumpDynamicMemPoolStateInfo() {}

  // Display the detailed debug information of memory block and memory buf.
  virtual void DumpDynamicMemPoolDebugInfo() {}

  // The statistics information.
  virtual size_t TotalMemStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t TotalUsedMemStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t TotalUsedByEventMemStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t TotalIdleMemStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t TotalEagerFreeMemStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t UsedMemPeakStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t MaxMemAllocatedStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t MaxMemReservedStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual size_t ActualPeakStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return 0;
  }

  virtual std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return {};
  }

  virtual std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return {};
  }

  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  CommonMemBlocksInfoStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return {};
  }

  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  PersistentMemBlocksInfoStatistics() const {
    RT_GLOG(ERROR) << "Not implemented";
    return {};
  }

  virtual void ResetMaxMemReserved() {
    RT_GLOG(ERROR) << "Not implemented";
  }

  virtual void ResetMaxMemAllocated() {
    RT_GLOG(ERROR) << "Not implemented";
  }

  virtual std::string GetMemoryPoolType() const {
    return "Other";
  }

  virtual const bool IsEnableEagerFree() const {
    return false;
  }

  virtual const bool IsEnableVmm() const {
    return false;
  }

  virtual void SetEnableVmm(bool enableVmm) {}

  virtual const bool SyncAllStreams() {
    return false;
  }

  virtual size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr* addr) {
    return 0;
  }

  virtual size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) {
    return 0;
  }

  virtual size_t MmapDeviceMem(size_t size, DeviceMemPtr addr) {
    return 0;
  }

  virtual const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() {
    return {0, 0};
  }

  virtual bool IsEnableTimeEvent() {
    return false;
  }

  virtual void SetEnableTimeEvent(bool enableTimeEvent) {}

  virtual void EnablePluggableAllocator(std::function<MallocFuncType> allocFn, std::function<FreeFuncType> freeFn) {}

  virtual void DisablePluggableAllocator() {}

  // Use set method to avoid performance decrease.
  void SetMemoryProfilerCallback(const std::function<void()>& memoryProfilerCallback) {
    memoryProfilerCallback_ = memoryProfilerCallback;
  }

  void SetMemoryMstxCallback(
      const std::function<void(void*, size_t)> memoryMallocMstxCallback,
      const std::function<void(void*)> memoryFreeMstxCallback) {
    memoryMallocMstxCallback_ = memoryMallocMstxCallback;
    memoryFreeMstxCallback_ = memoryFreeMstxCallback;
  }

  // Set rank id getter for memory pool to generate dump path.
  virtual void SetRankIdGetter(const std::function<size_t()>& rankIdGetter) {
    if (rankIdGetter != nullptr) {
      rankIdGetter_ = rankIdGetter;
    }
  }

  void SetPipelineCallback(const std::function<void()>& pipelineCallback) {
    pipelineCallback_ = pipelineCallback;
  }

 protected:
  std::function<void()> memoryProfilerCallback_{nullptr};
  std::function<size_t()> rankIdGetter_ = []() { return SIZE_MAX; };
  std::function<void()> pipelineCallback_{nullptr};
  std::function<void(void*, size_t)> memoryMallocMstxCallback_{nullptr};
  std::function<void(void*)> memoryFreeMstxCallback_{nullptr};
};

// Recording information for debugging the memory allocator.
struct FXRT_EXPORT AllocatorDebugInfo {
  std::string name_{"Unknown"};
  memory::mem_pool::MemType type_{memory::mem_pool::MemType::kOther};
  int inputIndex_{-1};
  int outputIndex_{-1};
  uint8_t runMode_{0};
};

class FXRT_EXPORT DynamicMemAllocatorDebugInfo {
 public:
  static AllocatorDebugInfo& GetDebugInfo() noexcept;

  // Set the debug info when memory alloc.
  static void SetDebugInfo(
      const std::string& name,
      memory::mem_pool::MemType type,
      int inputIndex = -1,
      int outputIndex = -1,
      uint8_t runMode = 0);

 private:
  DynamicMemAllocatorDebugInfo() = default;
  virtual ~DynamicMemAllocatorDebugInfo() = default;
  DynamicMemAllocatorDebugInfo(const DynamicMemAllocatorDebugInfo&) = delete;
  DynamicMemAllocatorDebugInfo& operator=(const DynamicMemAllocatorDebugInfo&) = delete;
};

using TaskIdOnStreamEvent = std::pair<int64_t, DeviceEventPtr>;
struct FXRT_EXPORT EventBase {
  // Record event on mem buf.
  bool RecordEvent(int64_t taskIdOnStream, uint32_t userStreamId, const DeviceEventPtr& event);

  // Release events on mem buf.
  bool WaitEvent(uint32_t taskIdOnStream, uint32_t userStreamId);

  // Indicates if mem buf used by event, return true when no event bind on mem buf.
  bool IsEventNotUsed();

  // Sync all events that bound on mem buf.
  bool SyncAllEvents();

  // Parameter: userStreamId, list of <taskIdOnStream, event>.
  std::shared_ptr<std::unordered_map<uint32_t, std::shared_ptr<std::list<TaskIdOnStreamEvent>>>> events_{nullptr};
};

struct FXRT_EXPORT JsonBuilder {
  JsonBuilder() {
    buffer_ << "{";
  }

  template <typename T>
  void Append(std::string key, T value) {
    buffer_ << "\"" << key << "\":" << value << ",";
  }

  std::string ToString() {
    buffer_.seekp(-1, buffer_.cur);
    buffer_ << "}";
    return buffer_.str();
  }

  std::stringstream buffer_;
};

struct FXRT_EXPORT MemoryTimeEvent {
  // Creation time of address in ns.
  uint64_t createdAt_{0};

  // Device address.
  void* addr_{nullptr};

  // Size of memory allocation.
  size_t size_{0};

  // Used size of memory pool.
  size_t usedSize_{0};

  // Peak size of memory pool.
  size_t peakSize_{0};

  // Allocate size of memory pool.
  size_t allocSize_{0};

  // Memory size that referred by event.
  size_t usedByEventSize_{0};

  // Eager free memory size.
  size_t eagerFreeSize_{0};

  // Whether allocation from persistent memory.
  uint8_t fromPersistent_{false};

  // Whether allocated memory is persistent.
  uint8_t isPersistent_{false};

  // pynative or graph or ge.
  uint8_t runMode_{0};

  // Data type of this address.
  uint8_t allocType_;

  // Stream id of address.
  uint32_t streamId_{0};

  // Owner of this address.
  std::string owner_;

  std::string ToJson() {
    JsonBuilder builder;
    builder.Append("createdAt_", createdAt_);
    builder.Append("addr_", addr_);
    builder.Append("size_", size_);
    builder.Append("fromPersistent_", fromPersistent_);
    builder.Append("streamId_", streamId_);
    builder.Append("runMode_", runMode_);
    builder.Append("usedSize_", usedSize_);
    builder.Append("peakSize_", peakSize_);
    builder.Append("allocSize_", allocSize_);
    builder.Append("usedByEventSize_", usedByEventSize_);
    builder.Append("eagerFreeSize_", eagerFreeSize_);
    builder.Append("owner_", owner_);
    builder.Append("allocType_", allocType_);
    return builder.ToString();
  }
};
using MemoryTimeEventPtr = std::shared_ptr<MemoryTimeEvent>;
} // namespace device
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_MEMORY_DYNAMIC_MEM_POOL_H_
