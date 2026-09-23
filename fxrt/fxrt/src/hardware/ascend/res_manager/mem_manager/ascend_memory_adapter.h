#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_ADAPTER_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_ADAPTER_H_

#include <algorithm>
#include <mutex>
#include <string>
#include <memory>
#include <vector>
#include <limits>

#include "common/common.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
namespace ascend {
struct MemoryBlock {
  MemoryBlock(void* ptr, const size_t size, const std::string& tag) {
    memPtr = ptr;
    memSize = size;
    memTag = tag;
  }

  void* memPtr{nullptr};
  size_t memSize{0};
  std::string memTag;
};

class AscendMemAdapter;
using AscendMemAdapterPtr = std::shared_ptr<AscendMemAdapter>;

class FXRT_EXPORT AscendMemAdapter {
 public:
  virtual ~AscendMemAdapter() = default;
  static AscendMemAdapterPtr GetInstance();

  virtual bool Initialize();
  virtual bool DeInitialize();

  virtual uint8_t* MallocStaticDevMem(size_t size, const std::string& tag = "") = 0;
  virtual bool FreeStaticDevMem(void* addr) = 0;
  virtual uint8_t* MallocDynamicDevMem(size_t size, const std::string& tag = "") = 0;
  virtual void ResetDynamicMemory() = 0;
  virtual std::string DevMemStatistics() const = 0;
  virtual size_t GetDynamicMemUpperBound(void* minStaticAddr) const = 0;
  [[nodiscard]] virtual uint64_t FreeDevMemSize() const = 0;

  virtual void SimulationInitialize();

  int64_t GetActualPeakMemory() const {
    return actualPeakMemory_;
  }
  int64_t GetUsedPeakMemory() const {
    return usedPeakMemory_;
  }
  void UpdateActualPeakMemory(int64_t memory) {
    actualPeakMemory_ = std::max(actualPeakMemory_, memory);
  }
  void UpdateUsedPeakMemory(int64_t memory) {
    usedPeakMemory_ = std::max(usedPeakMemory_, memory);
  }
  [[nodiscard]] uint64_t MaxHbmSizeForMs() const {
    return maxAvailableMsHbmSize_;
  }
  [[nodiscard]] int64_t GetMsUsedHbmSize() const {
    return msUsedHbmSize_;
  }
  static size_t GetRoundUpAlignSize(size_t inputSize);
  static size_t GetRoundDownAlignSize(size_t inputSize);

 protected:
  AscendMemAdapter() = default;
  uint8_t* MallocFromRts(size_t size) const;
  bool FreeToRts(void* devPtr, const size_t size) const;

  bool initialized_{false};
  // Support multi-thread.
  std::mutex mutex_;

  // Actual peak memory usage (with fragments)
  int64_t actualPeakMemory_{0};
  // Used peak memory usage (without fragments)
  int64_t usedPeakMemory_{0};

  // rts Memory INFO
  size_t deviceHbmTotalSize_{0};
  size_t deviceHbmFreeSize_{0};
  size_t deviceHbmHugePageReservedSize_{0};

  int64_t msUsedHbmSize_{0};
  int64_t maxAvailableMsHbmSize_{0};

 private:
  DISABLE_COPY_AND_ASSIGN(AscendMemAdapter)
  size_t GetDeviceMemSizeFromContext() const;
  static AscendMemAdapterPtr instance_;
};
} // namespace ascend
} // namespace device
} // namespace fxrt

#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_ADAPTER_H_
