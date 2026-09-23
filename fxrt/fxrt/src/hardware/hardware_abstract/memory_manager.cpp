#include "hardware/hardware_abstract/memory_manager.h"
#include <string>
#include "common/common.h"

namespace fxrt {
namespace device {
constexpr size_t kAlignBytes = 32;

size_t MemoryManager::GetCommonAlignSize(size_t inputSize) {
  return ((inputSize + kMemAlignSize + kAlignBytes - 1) / kMemAlignSize) * kMemAlignSize;
}

size_t MemoryManager::GetCommunicationAlignSize(size_t inputSize) {
  return ((inputSize + kMemAlignSize - 1) / kMemAlignSize) * kMemAlignSize + kTwiceMemAlignSize;
}

void MemoryManager::FreeMemFromMemPool(void* devicePtr) {
  if (devicePtr == nullptr) {
    RT_GLOG(ERROR) << "FreeMemFromMemPool devicePtr is null.";
  }
}

uint8_t* MemoryManager::MallocWorkSpaceMem(size_t size) {
  return MallocDynamicMem(size, false);
}

uint8_t* MemoryManager::MallocDynamicMem(size_t size, bool communicationMem) {
  RT_VLOG(VL_HARDWARE) << "Call default dynamic malloc " << size << " v " << communicationMem;
  return nullptr;
}

void* MemoryManager::MallocMemFromMemPool(size_t size, bool fromPersistentMem, bool, uint32_t streamId) {
  if (size == 0) {
    RT_GLOG(ERROR) << "MallocMemFromMemPool size is 0.";
  }
  return nullptr;
}

std::vector<void*> MemoryManager::MallocContinuousMemFromMemPool(
    const std::vector<size_t>& sizeList,
    uint32_t streamId) {
  if (sizeList.empty()) {
    RT_GLOG(ERROR) << "MallocContinuousMemFromMemPool size list's size is 0.";
  }
  std::vector<void*> devicePtrList;
  for (size_t i = 0; i < sizeList.size(); ++i) {
    (void)devicePtrList.emplace_back(nullptr);
  }
  return devicePtrList;
}
} // namespace device
} // namespace fxrt
