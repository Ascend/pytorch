#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_POOL_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_POOL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hardware/hardware_abstract/memory/abstract_dynamic_mem_pool.h"
#include "common/visible.h"
#include "hardware/ascend/res_manager/mem_manager/abstract_ascend_memory_pool_support.h"

namespace fxrt {
namespace device {
namespace ascend {

class FXRT_EXPORT DefaultAscendMemoryPool : public AbstractAscendMemoryPoolSupport,
                                            public AbstractEnhancedDynamicMemPool {
 public:
  DefaultAscendMemoryPool();
  DefaultAscendMemoryPool(const DefaultAscendMemoryPool&) = delete;
  DefaultAscendMemoryPool& operator=(const DefaultAscendMemoryPool&) = delete;
  ~DefaultAscendMemoryPool() override = default;

  std::string GetMemoryPoolType() const override {
    return "DefaultAscendMemoryPool";
  }

  void SetMemPoolBlockSize(size_t availableDeviceMemSize) override {
    return AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(availableDeviceMemSize);
  }

  size_t CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool needRecycle = false) override {
    return AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size, fromPersistentMem, needRecycle);
  }

  const bool IsEnableEagerFree() const override {
    return AbstractAscendMemoryPoolSupport::IsEnableEagerFree();
  }

  size_t EmptyCache() override;

  void EnablePluggableAllocator(std::function<MallocFuncType> allocFn, std::function<FreeFuncType> freeFn) override;
  void DisablePluggableAllocator() override;
};
using DefaultAscendMemoryPoolPtr = std::shared_ptr<DefaultAscendMemoryPool>;

class FXRT_EXPORT DefaultEnhancedAscendMemoryPool : public DefaultAscendMemoryPool {
 public:
  explicit DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr& instance);
  DefaultEnhancedAscendMemoryPool(const DefaultEnhancedAscendMemoryPool&) = delete;
  DefaultEnhancedAscendMemoryPool& operator=(const DefaultEnhancedAscendMemoryPool&) = delete;
  ~DefaultEnhancedAscendMemoryPool() override = default;

  // Wrap enhanced function.
  void Initialize(size_t initSize, size_t increaseSize, size_t maxSize) override {
    instance_->Initialize(initSize, increaseSize, maxSize);
  }

  void ReleaseDeviceRes() override;

  DeviceMemPtr AllocTensorMem(
      size_t size,
      bool fromPersistentMem = false,
      bool needRecycle = false,
      uint32_t streamId = kDefaultStreamIndex) override;

  std::vector<DeviceMemPtr> AllocContinuousTensorMem(
      const std::vector<size_t>& sizeList,
      uint32_t streamId = kDefaultStreamIndex) override;

  void FreeTensorMem(const DeviceMemPtr& deviceAddr) override;

  bool DoFreeTensorMem(const DeviceMemPtr& deviceAddr) override;

  void FreePartTensorMems(
      const std::vector<DeviceMemPtr>& freeAddrs,
      const std::vector<DeviceMemPtr>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) override;

  std::vector<MemBuf*> DoFreePartTensorMems(
      const std::vector<DeviceMemPtr>& freeAddrs,
      const std::vector<DeviceMemPtr>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) override {
    return instance_->DoFreePartTensorMems(freeAddrs, keepAddrs, keepAddrSizes);
  }

  void DefragMemory() override;

  void DumpDynamicMemPoolStateInfo() override;

  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() override;

  size_t ReleaseFreeBlocks() override {
    return instance_->ReleaseFreeBlocks();
  }

  // Proxy wrapper for AbstractAscendMemoryPoolSupport
  void ResetIdleMemBuf() const override {
    instance_->ResetIdleMemBuf();
  }

  bool RecordEvent(
      int64_t taskIdOnStream,
      uint32_t userStreamId,
      const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
      const DeviceEventPtr& event) override {
    return instance_->RecordEvent(taskIdOnStream, userStreamId, memoryStreamAddresses, event);
  }

  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) override;

  bool WaitEvent(int64_t taskIdOnStream, uint32_t memoryStreamId) override;

  bool SyncAllEvents() override;

  void EnablePluggableAllocator(std::function<MallocFuncType> allocFn, std::function<FreeFuncType> freeFn) override {
    return instance_->EnablePluggableAllocator(allocFn, freeFn);
  }

  void DisablePluggableAllocator() override {
    return instance_->DisablePluggableAllocator();
  }

  size_t AlignMemorySize(size_t size) const override {
    return instance_->AlignMemorySize(size);
  }

  size_t CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool needRecycle = false) override {
    return instance_->CalMemBlockAllocSize(size, fromPersistentMem, needRecycle);
  }

  void SetMemPoolBlockSize(size_t availableDeviceMemSize) override {
    instance_->SetMemPoolBlockSize(availableDeviceMemSize);
  }

  size_t MemAllocUnitSize(bool fromPersistentMem) const override {
    return instance_->MemAllocUnitSize(fromPersistentMem);
  }

  void SetMemAllocUintSize(size_t commonSize, size_t persistSize = kDynamicMemAllocUnitSize) override {
    instance_->SetMemAllocUintSize(commonSize, persistSize);
  }

  void* GetMinUsingMemoryAddr() const override {
    return instance_->GetMinUsingMemoryAddr();
  }

  size_t AllocDeviceMem(size_t size, DeviceMemPtr* addr) override {
    return instance_->AllocDeviceMem(size, addr);
  }

  bool FreeDeviceMem(const DeviceMemPtr& addr) override {
    return instance_->FreeDeviceMem(addr);
  }

  size_t free_mem_size() override {
    return instance_->free_mem_size();
  }

  uint64_t total_mem_size() const override {
    return instance_->total_mem_size();
  }

  size_t GetMaxUsedMemSize() const override {
    return instance_->GetMaxUsedMemSize();
  }

  size_t GetVmmUsedMemSize() const override {
    return instance_->GetVmmUsedMemSize();
  }

  void DumpDynamicMemPoolDebugInfo() override {
    instance_->DumpDynamicMemPoolDebugInfo();
  }

  size_t TotalMemStatistics() const override {
    return instance_->TotalMemStatistics();
  }

  size_t TotalUsedMemStatistics() const override {
    return instance_->TotalUsedMemStatistics();
  }

  size_t TotalUsedByEventMemStatistics() const override {
    return instance_->TotalUsedByEventMemStatistics();
  }

  size_t TotalIdleMemStatistics() const override {
    return instance_->TotalIdleMemStatistics();
  }

  size_t TotalEagerFreeMemStatistics() const override {
    return instance_->TotalEagerFreeMemStatistics();
  }

  size_t UsedMemPeakStatistics() const override {
    return instance_->UsedMemPeakStatistics();
  }

  size_t MaxMemAllocatedStatistics() const override {
    return instance_->MaxMemAllocatedStatistics();
  }

  size_t MaxMemReservedStatistics() const override {
    return instance_->MaxMemReservedStatistics();
  }

  size_t ActualPeakStatistics() const override {
    return instance_->ActualPeakStatistics();
  }

  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override {
    return std::move(instance_->BlockCountsStatistics());
  }

  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override {
    return std::move(instance_->BlockUnitSizeStatistics());
  }

  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
      const override {
    return std::move(instance_->CommonMemBlocksInfoStatistics());
  }

  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
      const override {
    return std::move(instance_->PersistentMemBlocksInfoStatistics());
  }

  void ResetMaxMemReserved() override {
    instance_->ResetMaxMemReserved();
  }

  void ResetMaxMemAllocated() override {
    instance_->ResetMaxMemAllocated();
  }

  const bool IsEnableEagerFree() const override {
    return instance_->IsEnableEagerFree();
  }

  const bool IsEnableVmm() const override {
    return instance_->IsEnableVmm();
  }

  void SetEnableVmm(bool enableVmm) override {
    instance_->SetEnableVmm(enableVmm);
  }

  const bool SyncAllStreams() override {
    return instance_->SyncAllStreams();
  }

  size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr* addr) override {
    return instance_->AllocDeviceMemByEagerFree(size, addr);
  }

  size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) override {
    return instance_->FreeDeviceMemByEagerFree(addr, size);
  }

  size_t MmapDeviceMem(size_t size, DeviceMemPtr addr) override {
    return instance_->MmapDeviceMem(size, addr);
  }

  std::string GetMemoryPoolType() const override {
    return "DefaultEnhancedAscendMemoryPool";
  }

  void ReportMemoryPoolInfo() override {
    instance_->ReportMemoryPoolInfo();
  }

  void ReportMemoryPoolMallocInfoToMstx(void* ptr, size_t size) override {
    instance_->ReportMemoryPoolMallocInfoToMstx(ptr, size);
  }

  void ReportMemoryPoolFreeInfoToMstx(void* ptr) override {
    instance_->ReportMemoryPoolFreeInfoToMstx(ptr);
  }

  bool IsEnableTimeEvent() override {
    return instance_->IsEnableTimeEvent();
  }

  void SetEnableTimeEvent(bool enableTimeEvent) override {
    instance_->SetEnableTimeEvent(enableTimeEvent);
  }

  MemoryTimeEventPtr GenAllocateMemoryTimeEvent(
      const void* addr,
      size_t size,
      uint32_t streamId,
      bool fromPersistent,
      bool isPersistent) override {
    return instance_->GenAllocateMemoryTimeEvent(addr, size, streamId, fromPersistent, isPersistent);
  }

  MemoryTimeEventPtr GenFreeMemoryTimeEvent(const void* addr) override {
    return instance_->GenFreeMemoryTimeEvent(addr);
  }

  size_t EmptyCache() override {
    return instance_->EmptyCache();
  }

 protected:
  void SetRankIdGetter(const std::function<size_t()>& rankIdGetter) override;

 private:
  DefaultAscendMemoryPoolPtr instance_;
  size_t lastVmmUsedSize_{0};
};

class FXRT_EXPORT BestFitAscendMemoryPool : public AbstractAscendMemoryPoolSupport {
 public:
  BestFitAscendMemoryPool();
  BestFitAscendMemoryPool(const BestFitAscendMemoryPool&) = delete;
  BestFitAscendMemoryPool& operator=(const BestFitAscendMemoryPool&) = delete;
  ~BestFitAscendMemoryPool() override = default;

  void SetMemPoolBlockSize(size_t availableDeviceMemSize) override {
    return AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(availableDeviceMemSize);
  }

  size_t CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool needRecycle = false) override {
    return AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size, fromPersistentMem, needRecycle);
  }

  const bool IsEnableEagerFree() const override {
    return AbstractAscendMemoryPoolSupport::IsEnableEagerFree();
  }

  std::string GetMemoryPoolType() const override {
    return "BestFitAscendMemoryPool";
  }

  size_t EmptyCache() override;
};

class FXRT_EXPORT AscendMemoryPool {
 public:
  AscendMemoryPool(const AscendMemoryPool&) = delete;
  AscendMemoryPool& operator=(const AscendMemoryPool&) = delete;

  static AbstractAscendMemoryPoolSupport& GetInstance();

  static void SetEnhancedMemoryPool(bool enable);

 private:
  AscendMemoryPool() {}

  static bool UseOldMemoryPool();

  // Use enhanced memory pool when enable debug, enable log, enable prof, dry run and so on.
  static bool UseEnhancedMemoryPool();

  // Reference to memory pool.
  static AbstractAscendMemoryPoolSupportPtr pool_;

  // Basic memory pool instance with high performance.
  static AbstractAscendMemoryPoolSupportPtr instance_;

  // Memory pool support profiling and debugging.
  static AbstractAscendMemoryPoolSupportPtr enhancedInstance_;
};
} // namespace ascend
} // namespace device
} // namespace fxrt

#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_POOL_H_
