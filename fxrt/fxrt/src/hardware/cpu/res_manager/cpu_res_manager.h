#ifndef FXRT_SRC_HARDWARE_CPU_CPU_RES_MANAGER_H_
#define FXRT_SRC_HARDWARE_CPU_CPU_RES_MANAGER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "hardware/hardware_abstract/device_context.h"
#include "common/common.h"

namespace fxrt {
namespace device {
namespace cpu {
class FXRT_EXPORT CPUResManager : public DeviceResManager {
 public:
  CPUResManager() {
    Initialize();
  }
  ~CPUResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  // Relevant function to allocate and free device memory of raw ptr.
  void* AllocateMemory(size_t size, uint32_t streamId = kDefaultStreamIndex) const override;
  void FreeMemory(void* ptr) const override;
  void FreePartMemorys(
      const std::vector<void*>& freeAddrs,
      const std::vector<void*>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) const override;
  bool AsyncCopy(void* dst, const void* src, uint64_t size, CopyType kind, void* stream) const override;
  bool SyncCopy(void* dst, const void* src, uint64_t size, CopyType kind) const override;
};
} // namespace cpu
} // namespace device
} // namespace fxrt
#endif
