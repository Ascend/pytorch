#ifndef FXRT_SRC_HARDWARE_ASCEND_ABSTRACT_ASCEND_MEMORY_POOL_SUPPORT_H_
#define FXRT_SRC_HARDWARE_ASCEND_ABSTRACT_ASCEND_MEMORY_POOL_SUPPORT_H_

#include <memory>

#include "hardware/hardware_abstract/memory/dynamic_mem_pool.h"
#include "common/visible.h"

namespace fxrt {
namespace device {
namespace ascend {
// Definition for abstract ascend memory pool support class, wrap device interface of ascend.
class FXRT_EXPORT AbstractAscendMemoryPoolSupport : virtual public DynamicMemPool {
 public:
  ~AbstractAscendMemoryPoolSupport() override = default;

  size_t AllocDeviceMem(size_t size, DeviceMemPtr* addr) override;

  bool FreeDeviceMem(const DeviceMemPtr& addr) override;

  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr) override;

  size_t GetMaxUsedMemSize() const override;

  size_t GetVmmUsedMemSize() const override;

  size_t free_mem_size() override;

  uint64_t total_mem_size() const override;

  // Set mem pool block size
  void SetMemPoolBlockSize(size_t availableDeviceMemSize) override;

  virtual void ResetIdleMemBuf() const;

  // Calculate memory block required alloc size when adding the memory block.
  size_t CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool needRecycle) override;

  // The related interface of device memory eager free.
  const bool IsEnableEagerFree() const override;

  const bool SyncAllStreams() override;

  size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr* addr) override;

  size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) override;

  size_t EmptyCache() override;
};
using AbstractAscendMemoryPoolSupportPtr = std::shared_ptr<AbstractAscendMemoryPoolSupport>;
} // namespace ascend
} // namespace device
} // namespace fxrt

#endif // #define FXRT_SRC_HARDWARE_ASCEND_ABSTRACT_ASCEND_MEMORY_POOL_SUPPORT_H_
