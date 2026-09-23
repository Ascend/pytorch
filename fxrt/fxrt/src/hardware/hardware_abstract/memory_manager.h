#ifndef FXRT_SRC_HARDWARE_MEMORY_MANAGER_H_
#define FXRT_SRC_HARDWARE_MEMORY_MANAGER_H_
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include "common/logger.h"
#include "hardware/hardware_abstract/memory/dynamic_mem_pool.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
enum class MemType { kStaticMem, kDynamicMem, kSomasReuseDynamicMem };
const uint32_t kInvalidGraphId = UINT32_MAX;
constexpr int kGetAllOuts = -1;
constexpr uint64_t kMemAlignSize = 512;
constexpr uint64_t kTwiceMemAlignSize = kMemAlignSize << 1;
class FXRT_EXPORT MemoryManager {
 public:
  MemoryManager() = default;
  virtual ~MemoryManager() = default;

  virtual void Initialize() = 0;
  virtual void Finalize() = 0;
  virtual void ResetDynamicMemory() {}
  virtual void ClearGlobalIdleMem() {}

  uint8_t* MallocWorkSpaceMem(size_t size);
  virtual void* MallocMemFromMemPool(
      size_t size,
      bool fromPersistentMem,
      bool needRecycle = false,
      uint32_t streamId = kDefaultStreamIndex);
  virtual size_t GetMaxUsedMemorySize() const {
    return 0;
  }
  virtual void FreeMemFromMemPool(void* devicePtr);
  virtual std::vector<void*> MallocContinuousMemFromMemPool(
      const std::vector<size_t>& sizeList,
      uint32_t streamId = kDefaultStreamIndex);

  static size_t GetCommonAlignSize(size_t inputSize);
  static size_t GetCommunicationAlignSize(size_t inputSize);

  virtual size_t GetAvailableMemSize() {
    RT_GLOG(ERROR) << "Return default 0 mem size!";
    return 0;
  }

  bool RecordEvent(
      int64_t taskIdOnStream,
      uint32_t userStreamId,
      const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
      const DeviceEventPtr& event) {
    if (GetMemoryPool() == nullptr) {
      RT_VLOG(VL_HARDWARE) << "memory pool is nullptr.";
      return false;
    }
    return GetMemoryPool()->RecordEvent(taskIdOnStream, userStreamId, memoryStreamAddresses, event);
  }
  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) {
    if (GetMemoryPool() == nullptr) {
      RT_VLOG(VL_HARDWARE) << "memory pool is nullptr.";
      return false;
    }
    return GetMemoryPool()->WaitEvent(taskIdOnStream, userStreamId, memoryStreamId);
  }
  bool WaitEvent(int64_t taskIdOnStream, uint32_t memoryStreamId) {
    if (GetMemoryPool() == nullptr) {
      RT_VLOG(VL_HARDWARE) << "memory pool is nullptr.";
      return false;
    }
    return GetMemoryPool()->WaitEvent(taskIdOnStream, memoryStreamId);
  }
  bool SyncAllEvents() {
    if (GetMemoryPool() == nullptr) {
      RT_VLOG(VL_HARDWARE) << "memory pool is nullptr.";
      return false;
    }
    return GetMemoryPool()->SyncAllEvents();
  }

  virtual DynamicMemPool* GetMemoryPool() = 0;

  // Relevant function to manage memory statistics
  virtual size_t GetTotalMemStatistics() const {
    return 0;
  }
  virtual size_t GetTotalUsedMemStatistics() const {
    return 0;
  }
  virtual size_t GetTotalIdleMemStatistics() const {
    return 0;
  }
  virtual size_t GetTotalEagerFreeMemStatistics() const {
    return 0;
  }
  virtual size_t GetUsedMemPeakStatistics() const {
    return 0;
  }
  virtual size_t GetReservedMemPeakStatistics() const {
    return 0;
  }
  virtual std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const {
    return {};
  }
  virtual std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const {
    return {};
  }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetCommonMemBlocksInfoStatistics() const {
    return {};
  }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const {
    return {};
  }
  virtual void ResetMaxMemoryReserved() {}
  virtual void ResetMaxMemoryAllocated() {}
  virtual size_t EmptyCache() {
    return -1L;
  }

 protected:
  virtual uint8_t* MallocStaticMem(size_t size, bool communicationMem, uint32_t graphId) = 0;
  virtual uint8_t* MallocStaticMem(size_t size, bool communicationMem) {
    return MallocStaticMem(size, communicationMem, kInvalidGraphId);
  }
  virtual uint8_t* MallocDynamicMem(size_t size, bool communicationMem);

  // Hold memory pool for common operations on memory.
  DynamicMemPool* memoryPool_{nullptr};
};
} // namespace device
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_MEMORY_MANAGER_H_
