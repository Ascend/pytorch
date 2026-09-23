#include "hardware/ascend/res_manager/mem_manager/abstract_ascend_memory_pool_support.h"

#include <algorithm>
#include <utility>

#include "hardware/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "common/common.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace fxrt {
namespace device {
namespace ascend {
// The minimum unit size (8MB) of memory block used for dynamic extend in graph run mode.
static const size_t ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE = 8 << 20;
constexpr char kGlobalOverflowWorkspace[] = "GLOBAL_OVERFLOW_WORKSPACE";

void AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(size_t availableDeviceMemSize) {
  // set by default configuration
  SetMemAllocUintSize(kDynamicMemAllocUnitSize, kDynamicMemAllocUnitSize);
}

namespace {
bool NoAdditionalMemory() {
  // use default temporarily.
  return true;
}
} // namespace

size_t AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool needRecycle) {
  auto deviceFreeMemSize = free_mem_size();
  if (deviceFreeMemSize < size) {
    RT_VLOG(VL_HARDWARE) << "The device memory is not enough, the free memory size is " << deviceFreeMemSize
                         << ", but the alloc size is " << size;
    RT_VLOG(VL_HARDWARE) << "The dynamic memory pool total size is " << TotalMemStatistics() / kMBToByte
                         << "M, total used size is " << TotalUsedMemStatistics() / kMBToByte << "M, used peak size is "
                         << UsedMemPeakStatistics() / kMBToByte << "M.";
    RT_VLOG(VL_HARDWARE) << "Memory Statistics:" << AscendMemAdapter::GetInstance()->DevMemStatistics();
    return 0;
  }

  size_t allocMemSize;
  SetMemPoolBlockSize(deviceFreeMemSize);
  auto allocMemUnitSize = MemAllocUnitSize(fromPersistentMem);
  if (needRecycle) {
    allocMemUnitSize = kDynamicMemAllocUnitSize;
  }
  RT_VLOG(VL_HARDWARE) << "Get unit block size " << allocMemUnitSize;
  allocMemSize = allocMemUnitSize;

  const bool isGraphRunMode = true;
  // cppcheck-suppress knownConditionTrueFalse
  if (isGraphRunMode) {
    // Growing at adding alloc unit size
    while (allocMemSize < size) {
      allocMemSize = allocMemSize + allocMemUnitSize;
    }
  } else {
    // Growing at twice of alloc unit size
    constexpr size_t kDouble = 2;
    while (allocMemSize < size) {
      allocMemSize = allocMemSize * kDouble;
    }
  }

  allocMemSize = std::min(allocMemSize, deviceFreeMemSize);
  if (NoAdditionalMemory() && !needRecycle) {
    allocMemSize = std::min(allocMemSize, size);
  }
  return allocMemSize;
}

size_t AbstractAscendMemoryPoolSupport::AllocDeviceMem(size_t size, DeviceMemPtr* addr) {
  RT_VLOG(VL_HARDWARE) << "Malloc Memory for Pool, size: " << size;
  if (size == 0) {
    RT_GLOG(ERROR) << "Failed to alloc memory pool resource, the size is zero!";
  }
  *addr = AscendMemAdapter::GetInstance()->MallocStaticDevMem(size);
  if (*addr == nullptr) {
    RT_GLOG(ERROR) << "Alloc device memory pool address is nullptr, failed to alloc memory pool resource!";
  }
  return size;
}

size_t AbstractAscendMemoryPoolSupport::GetMaxUsedMemSize() const {
  void* minUsedAddr = GetMinUsingMemoryAddr();
  if (minUsedAddr == nullptr) {
    return 0;
  }
  return AscendMemAdapter::GetInstance()->GetDynamicMemUpperBound(minUsedAddr);
}

size_t AbstractAscendMemoryPoolSupport::GetVmmUsedMemSize() const {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().GetAllocatedSize();
  }
  return 0;
}

const bool AbstractAscendMemoryPoolSupport::IsEnableEagerFree() const {
  return AscendGmemAdapter::GetInstance().is_eager_free_enabled();
}

const bool AbstractAscendMemoryPoolSupport::SyncAllStreams() {
  return AscendStreamMng::GetInstance().SyncAllStreams();
}

size_t AbstractAscendMemoryPoolSupport::AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr* addr) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().AllocDeviceMem(size, addr);
  } else if (IsEnableEagerFree()) {
    return AscendGmemAdapter::GetInstance().AllocDeviceMem(size, addr);
  } else {
    RT_GLOG(ERROR) << "Eager free and VMM are both disabled.";
    return 0;
  }
}

size_t AbstractAscendMemoryPoolSupport::FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().EagerFreeDeviceMem(addr, size);
  } else if (IsEnableEagerFree()) {
    return AscendGmemAdapter::GetInstance().EagerFreeDeviceMem(addr, size);
  } else {
    RT_GLOG(ERROR) << "Eager free and VMM are both disabled.";
    return 0;
  }
}

size_t AbstractAscendMemoryPoolSupport::EmptyCache() {
  return AscendVmmAdapter::GetInstance().EmptyCache();
}

size_t AbstractAscendMemoryPoolSupport::MmapDeviceMem(const size_t size, const DeviceMemPtr addr) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().MmapDeviceMem(size, addr, total_mem_size());
  } else if (IsEnableEagerFree()) {
    auto ret = AscendGmemAdapter::GetInstance().MmapMemory(size, addr);
    if (ret == nullptr) {
      RT_GLOG(ERROR) << "Mmap memory failed.";
    }
    return size;
  }
  RT_GLOG(ERROR) << "Eager free and VMM are both disabled.";
  return 0;
}

bool AbstractAscendMemoryPoolSupport::FreeDeviceMem(const DeviceMemPtr& addr) {
  CHECK_IF_NULL(addr);
  int64_t maxActual = ActualPeakStatistics();
  RT_VLOG(VL_HARDWARE) << "Max actual used memory size is " << maxActual;
  AscendMemAdapter::GetInstance()->UpdateActualPeakMemory(maxActual);
  int64_t maxPeak = UsedMemPeakStatistics();
  RT_VLOG(VL_HARDWARE) << "Max peak used memory size is " << maxPeak;
  AscendMemAdapter::GetInstance()->UpdateUsedPeakMemory(maxPeak);
  // disable ge kernel use two pointer mem adapter, not support free.
  // if (!IsEnableVmm() && !IsEnableEagerFree() && !IsDisableGeKernel()) {
  //   return AscendMemAdapter::GetInstance()->FreeStaticDevMem(addr);
  // }
  return true;
}

void AbstractAscendMemoryPoolSupport::ResetIdleMemBuf() const {
  // Warning : This method is not in used currently, removed in next release.
}

size_t AbstractAscendMemoryPoolSupport::free_mem_size() {
  return AscendMemAdapter::GetInstance()->FreeDevMemSize();
}

uint64_t AbstractAscendMemoryPoolSupport::total_mem_size() const {
  return AscendMemAdapter::GetInstance()->MaxHbmSizeForMs();
}
} // namespace ascend
} // namespace device
} // namespace fxrt
