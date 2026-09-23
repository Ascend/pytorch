#include "hardware/ascend/res_manager/mem_manager/ascend_dynamic_mem_adapter.h"
#include <algorithm>
#include <set>
#include "common/common.h"

#include "hardware/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace fxrt {
namespace device {
namespace ascend {
constexpr size_t kMBToByte = 1024 << 10;

uint8_t* AscendDynamicMemAdapter::MallocStaticDevMem(size_t size, const std::string& tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  if (hasAllocSize + size > LongToSize(maxAvailableMsHbmSize_)) {
    RT_GLOG(ERROR) << "No enough memory to allocate, hasAllocSize:" << hasAllocSize << ", size:" << size
                   << ", max_available_ms_moc_size:" << maxAvailableMsHbmSize_;
  }
  auto addr = MallocFromRts(size);
  if (addr != nullptr) {
    hasAllocSize += size;
    (void)staticMemoryBlocks_.emplace(addr, std::make_shared<MemoryBlock>(addr, size, tag));
    RT_VLOG(VL_HARDWARE) << "MallocStaticDevMem success, size:" << size << ", tag:" << tag;
  }
  return addr;
}

bool AscendDynamicMemAdapter::FreeStaticDevMem(void* addr) {
  RT_VLOG(VL_HARDWARE) << "FreeStaticDevMem addr:" << addr << ".";
  std::lock_guard<std::mutex> locker(mutex_);
  if (addr == nullptr) {
    RT_GLOG(ERROR) << "addr is nullptr.";
    return false;
  }
  auto&& iter = staticMemoryBlocks_.find(addr);
  if (iter == staticMemoryBlocks_.end()) {
    RT_GLOG(ERROR) << "addr is not in static memory blocks, addr:" << addr << ".";
    return false;
  }
  auto memBlock = iter->second;
  auto ret = FreeToRts(memBlock->memPtr, memBlock->memSize);
  if (!ret) {
    RT_GLOG(ERROR) << "Free memory failed.";
    return false;
  }
  RT_VLOG(VL_HARDWARE) << "Free memory success, addr:" << addr << ", size:" << memBlock->memSize << ".";
  hasAllocSize -= memBlock->memSize;
  staticMemoryBlocks_.erase(addr);
  return true;
}

bool AscendDynamicMemAdapter::Initialize() {
  if (initialized_) {
    return true;
  }
  (void)AscendMemAdapter::Initialize();
  initialized_ = true;
  RT_VLOG(VL_HARDWARE) << "Ascend Memory Adapter initialize success, Memory Statistics:" << DevMemStatistics();
  return true;
}

bool AscendDynamicMemAdapter::DeInitialize() {
  for (const auto& [addr, blk] : staticMemoryBlocks_) {
    if (blk->memPtr != nullptr) {
      auto ret = FreeToRts(blk->memPtr, blk->memSize);
      if (!ret) {
        RT_GLOG(ERROR) << "Free memory failed.";
        return false;
      }
      RT_VLOG(VL_HARDWARE) << "Free memory success, addr:" << addr << ", size:" << blk->memSize
                           << ", tag:" << blk->memTag;
    }
  }
  (void)AscendMemAdapter::DeInitialize();
  hasAllocSize = 0;
  staticMemoryBlocks_.clear();
  initialized_ = false;
  return true;
}

uint64_t AscendDynamicMemAdapter::FreeDevMemSize() const {
  return maxAvailableMsHbmSize_ - hasAllocSize;
}

uint8_t* AscendDynamicMemAdapter::MallocDynamicDevMem(size_t size, const std::string&) {
  RT_GLOG(ERROR) << "MallocDynamicDevMem is disabled.";
  return nullptr;
}

void AscendDynamicMemAdapter::ResetDynamicMemory() {
  RT_GLOG(ERROR) << "ResetDynamicMemory is disabled.";
}

std::string AscendDynamicMemAdapter::DevMemStatistics() const {
  std::ostringstream oss;
  oss << "\nDevice MOC memory size: " << deviceHbmTotalSize_ / kMBToByte << "M";
  oss << "\nfxrt Used memory size: " << msUsedHbmSize_ / kMBToByte << "M";
  auto printActualPeakMemory = AscendVmmAdapter::GetInstance().IsEnabled()
      ? AscendVmmAdapter::GetInstance().GetAllocatedSize()
      : actualPeakMemory_;
  oss << "\nUsed peak memory usage (without fragments): " << usedPeakMemory_ / kMBToByte << "M";
  oss << "\nActual peak memory usage (with fragments): " << printActualPeakMemory / kMBToByte << "M";
  oss << std::endl;
  return oss.str();
}

size_t AscendDynamicMemAdapter::GetDynamicMemUpperBound(void* minStaticAddr) const {
  RT_GLOG(ERROR) << "GetDynamicMemUpperBound is disabled.";
  return 0;
}
} // namespace ascend
} // namespace device
} // namespace fxrt
