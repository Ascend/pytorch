#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_DYNAMIC_MEM_ADAPTER_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_DYNAMIC_MEM_ADAPTER_H_

#include "hardware/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include <string>
#include <map>
#include <memory>

namespace fxrt {
namespace device {
namespace ascend {
class AscendDynamicMemAdapter : public AscendMemAdapter {
 public:
  bool Initialize() override;
  bool DeInitialize() override;
  uint8_t* MallocStaticDevMem(size_t size, const std::string& tag = "") override;
  bool FreeStaticDevMem(void* addr) override;
  uint8_t* MallocDynamicDevMem(size_t size, const std::string& tag = "") override;
  void ResetDynamicMemory() override;
  std::string DevMemStatistics() const override;
  size_t GetDynamicMemUpperBound(void* minStaticAddr) const override;
  [[nodiscard]] uint64_t FreeDevMemSize() const override;

 private:
  size_t hasAllocSize = 0;
  std::map<void*, std::shared_ptr<MemoryBlock>> staticMemoryBlocks_;
};
} // namespace ascend
} // namespace device
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_DYNAMIC_MEM_ADAPTER_H_
