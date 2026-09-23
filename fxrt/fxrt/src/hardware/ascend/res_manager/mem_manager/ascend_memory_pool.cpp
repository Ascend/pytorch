#include "hardware/ascend/res_manager/mem_manager/ascend_memory_pool.h"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <set>

#include "common/common.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"

namespace fxrt {
namespace device {
namespace ascend {
constexpr size_t kByteOffset = 8;
constexpr uint32_t kDefaultDispatchThreadsNum = 5;
constexpr uint32_t kDefaultOpThreadsNum = 25;
constexpr float kDefaultMemInitSize = 2.0;
constexpr float kDefaultMemBlockIncreaseSize = 1.0;
constexpr float kDefaultMemMaxSize = 1024.0;

DefaultAscendMemoryPool::DefaultAscendMemoryPool() {
  RT_VLOG(VL_HARDWARE) << "DefaultAscendMemoryPool constructed.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

size_t DefaultAscendMemoryPool::EmptyCache() {
  LockGuard lock(AbstractDynamicMemPool::lock());
  AbstractEnhancedDynamicMemPool::WaitPipelineHelper();
  AbstractAscendMemoryPoolSupport::SyncAllStreams();
  size_t releaseFreeSize = 0;
  if (FXRT_UNLIKELY(!customizedAllocators_.empty())) {
    releaseFreeSize += ReleaseCustomFreeBlocks();
  }
  if (IsEnableVmm()) {
    AbstractEnhancedDynamicMemPool::FreeIdleMemsByEagerFree();
    releaseFreeSize += AbstractAscendMemoryPoolSupport::EmptyCache();
    return releaseFreeSize;
  } else if (IsEnableEagerFree()) {
    auto ret = AbstractEnhancedDynamicMemPool::FreeIdleMemsByEagerFree();
    RT_VLOG(VL_HARDWARE) << "Eager free memory size is " << ret.second << ".";
    releaseFreeSize += ret.second;
    return releaseFreeSize;
  }

  RT_VLOG(VL_HARDWARE) << "Vmm is not enabled, try to release free blocks.";
  // // disable ge kernel use two pointer mem adapter, not support free.
  // if (IsDisableGeKernel()) {
  //   return 0L;
  // }
  releaseFreeSize += ReleaseFreeBlocks();
  return releaseFreeSize;
}

void DefaultAscendMemoryPool::EnablePluggableAllocator(
    std::function<MallocFuncType> allocFn,
    std::function<FreeFuncType> freeFn) {
  customAllocFn_ = allocFn;
  customFreeFn_ = freeFn;
  enableCustomAllocator_ = true;
}

void DefaultAscendMemoryPool::DisablePluggableAllocator() {
  enableCustomAllocator_ = false;
  return;
}

DefaultEnhancedAscendMemoryPool::DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr& instance)
    : instance_(instance) {
  RT_VLOG(VL_HARDWARE) << "DefaultEnhancedAscendMemoryPool constructed.";
  instance_->SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

void DefaultEnhancedAscendMemoryPool::ReleaseDeviceRes() {
  RT_VLOG(VL_HARDWARE) << "Start release device res.";
  instance_->ReleaseDeviceRes();
}

DeviceMemPtr DefaultEnhancedAscendMemoryPool::AllocTensorMem(
    size_t size,
    bool fromPersistentMem,
    bool needRecycle,
    uint32_t streamId) {
  size_t alignSize = AlignMemorySize(size);
  RT_VLOG(VL_HARDWARE) << "Allocate tensor mem, size : " << size << ", alignSize : " << alignSize
                       << ", needRecycle : " << needRecycle << ".";
  LockGuard lock(instance_->lock());
  const auto [memBuf, allocator] = instance_->AllocMemBuf(alignSize, fromPersistentMem, streamId);
  if (memBuf == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Allocate tensor mem, return nullptr.";
    // Dump mem pool state info and debug info when alloc tensor failed.
    DumpDynamicMemPoolStateInfo();
    DumpDynamicMemPoolDebugInfo();
    return nullptr;
  }

  memBuf->SetDebugInfo();
  instance_->addr_mem_buf_allocators().emplace(memBuf->addr_, std::make_pair(memBuf, allocator));
  auto deviceAddr = memBuf->addr_;

  instance_->ReportMemoryPoolInfo();
  instance_->ReportMemoryPoolMallocInfoToMstx(deviceAddr, alignSize);

  RT_VLOG(VL_HARDWARE) << "Allocate tensor mem, return : " << memBuf->ToJson()
                       << ", stat info : " << instance_->mem_stat().ToJson() << ".";
  return deviceAddr;
}

std::vector<DeviceMemPtr> DefaultEnhancedAscendMemoryPool::AllocContinuousTensorMem(
    const std::vector<size_t>& sizeList,
    uint32_t streamId) {
  RT_VLOG(VL_HARDWARE) << "Alloc continuous tensor mem, stream id : " << streamId << ".";
  const auto& continuousAddrs = instance_->AllocContinuousTensorMem(sizeList, streamId);
  if (continuousAddrs.size() != sizeList.size()) {
    return continuousAddrs;
  }
  if (continuousAddrs.size() == 1 && continuousAddrs[0] == nullptr) {
    return continuousAddrs;
  }
  return continuousAddrs;
}

void DefaultEnhancedAscendMemoryPool::FreeTensorMem(const DeviceMemPtr& deviceAddr) {
  RT_VLOG(VL_HARDWARE) << "Free tensor mem, device addr : " << deviceAddr << ".";
  LockGuard lock(instance_->lock());
  DoFreeTensorMem(deviceAddr);
}

bool DefaultEnhancedAscendMemoryPool::DoFreeTensorMem(const DeviceMemPtr& deviceAddr) {
  void* enhancedDeviceAddr = deviceAddr;
  bool ret = instance_->DoFreeTensorMem(deviceAddr);
  RT_VLOG(VL_HARDWARE) << "Do free tensor mem : " << enhancedDeviceAddr << ", return : " << ret << ".";
  return ret;
}

void DefaultEnhancedAscendMemoryPool::FreePartTensorMems(
    const std::vector<DeviceMemPtr>& freeAddrs,
    const std::vector<DeviceMemPtr>& keepAddrs,
    const std::vector<size_t>& keepAddrSizes) {
  RT_VLOG(VL_HARDWARE) << "Free part tensor mems.";
  LockGuard lock(instance_->lock());

  (void)instance_->DoFreePartTensorMems(freeAddrs, keepAddrs, keepAddrSizes);
}

void DefaultEnhancedAscendMemoryPool::DefragMemory() {
  if (lastVmmUsedSize_ == 0) {
    lastVmmUsedSize_ = GetVmmUsedMemSize();
  } else {
    size_t vmmUsedSize = GetVmmUsedMemSize();
    if (vmmUsedSize > lastVmmUsedSize_) {
      RT_VLOG(VL_HARDWARE) << "Current vmm used size : " << vmmUsedSize
                           << " is bigger than last vmm used size : " << lastVmmUsedSize_ << ".";
      lastVmmUsedSize_ = vmmUsedSize;
    }
  }

  instance_->DefragMemory();
}

void DefaultEnhancedAscendMemoryPool::DumpDynamicMemPoolStateInfo() {
  instance_->DumpDynamicMemPoolStateInfo();
}

const std::pair<size_t, size_t> DefaultEnhancedAscendMemoryPool::FreeIdleMemsByEagerFree() {
  const auto [eagerFreeSize, realFreeSize] = instance_->FreeIdleMemsByEagerFree();
  return {eagerFreeSize, realFreeSize};
}

bool DefaultEnhancedAscendMemoryPool::WaitEvent(
    int64_t taskIdOnStream,
    uint32_t userStreamId,
    uint32_t memoryStreamId) {
  LockGuard lock(instance_->lock());
  auto key = std::make_pair(userStreamId, memoryStreamId);
  auto iter = instance_->streamPairMemBufs().find(key);
  if (iter == instance_->streamPairMemBufs().end()) {
    return false;
  }

  auto memBufs_ = iter->second;
  for (const auto& memBuf : memBufs_) {
    RT_VLOG(VL_HARDWARE) << "Wait event for : " << memBuf->ToJson() << ".";
    memBuf->WaitEvent(taskIdOnStream, userStreamId);
    // Remove event and try to free memory.
    if (memBuf->IsEventNotUsed()) {
      instance_->mem_stat().usedByEventSize_ -= memBuf->size_;
      // Force clear all mem bufs.
      for (auto& streamPairMemBufs : instance_->streamPairMemBufs()) {
        (void)streamPairMemBufs.second.erase(memBuf);
      }
      if (memBuf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        (void)DoFreeTensorMem(memBuf->addr_);
      }
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::WaitEvent(int64_t taskIdOnStream, uint32_t memoryStreamId) {
  LockGuard lock(instance_->lock());
  for (auto& streamPairMemBufs : instance_->streamPairMemBufs()) {
    const auto& [userStream, memoryStream] = streamPairMemBufs.first;
    if (memoryStream != memoryStreamId) {
      continue;
    }
    auto memBufs = streamPairMemBufs.second;
    for (const auto& memBuf : memBufs) {
      RT_VLOG(VL_HARDWARE) << "Wait event for : " << memBuf->ToJson() << ".";
      memBuf->WaitEvent(taskIdOnStream, userStream);
      // Remove event and try to free memory.
      if (memBuf->IsEventNotUsed()) {
        instance_->mem_stat().usedByEventSize_ -= memBuf->size_;
        // Force clear all mem bufs.
        for (auto& kv : instance_->streamPairMemBufs()) {
          (void)kv.second.erase(memBuf);
        }
        if (memBuf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
          (void)DoFreeTensorMem(memBuf->addr_);
        }
      }
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::SyncAllEvents() {
  LockGuard lock(instance_->lock());
  if (streamPairMemBufs().empty()) {
    return false;
  }

  std::set<MemBuf*> carryEventMemBufs;
  for (const auto& streamPairMemBuf : instance_->streamPairMemBufs()) {
    for (const auto& memBuf : streamPairMemBuf.second) {
      (void)carryEventMemBufs.emplace(memBuf);
    }
  }
  for (auto& memBuf : carryEventMemBufs) {
    if (memBuf->SyncAllEvents() && memBuf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      (void)DoFreeTensorMem(memBuf->addr_);
    }
  }

  instance_->streamPairMemBufs().clear();
  return true;
}

void DefaultEnhancedAscendMemoryPool::SetRankIdGetter(const std::function<size_t()>& rankIdGetter) {
  instance_->SetRankIdGetter(rankIdGetter);
  if (rankIdGetter != nullptr) {
    rankIdGetter_ = rankIdGetter;
  }
}

BestFitAscendMemoryPool::BestFitAscendMemoryPool() {
  RT_VLOG(VL_HARDWARE) << "BestFitAscendMemoryPool constructed, older memory allocator is enabled.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

size_t BestFitAscendMemoryPool::EmptyCache() {
  RT_VLOG(VL_HARDWARE) << "Best fit memory pool is not supported empty cache.";
  return 0L;
}

// Initialize static member in AscendMemoryPool.
AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::pool_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::instance_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::enhancedInstance_ = nullptr;

AbstractAscendMemoryPoolSupport& AscendMemoryPool::GetInstance() {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    if (UseOldMemoryPool()) {
      instance_ = std::make_shared<BestFitAscendMemoryPool>();
      enhancedInstance_ = instance_;
    } else {
      const auto& memory_pool = std::make_shared<DefaultAscendMemoryPool>();
      instance_ = memory_pool;
      enhancedInstance_ = std::make_shared<DefaultEnhancedAscendMemoryPool>(memory_pool);
    }
    // Initialize instance and set ptr.
    float initSize = kDefaultMemInitSize;
    size_t initSizeByte = FloatToSize(initSize * kGBToByte);
    float increaseSize = kDefaultMemBlockIncreaseSize;
    size_t increaseSizeByte = FloatToSize(increaseSize * kGBToByte);
    float maxSize = kDefaultMemMaxSize;
    size_t maxSizeByte = FloatToSize(maxSize * kGBToByte);
    instance_->Initialize(initSizeByte, increaseSizeByte, maxSizeByte);
    // Set memory mstx callback func.
    if (!UseEnhancedMemoryPool()) {
      pool_ = instance_;
    } else {
      pool_ = enhancedInstance_;
    }
  });
  return *pool_;
}

void AscendMemoryPool::SetEnhancedMemoryPool(bool enable) {
  RT_VLOG(VL_HARDWARE) << "Set enhanced memory pool : " << enable << ".";
  if (enable) {
    pool_ = enhancedInstance_;
  } else {
    pool_ = instance_;
  }
}

bool AscendMemoryPool::UseOldMemoryPool() {
  return false;
  // if (memory::mem_pool::IsDisableAllocConfig(memory::mem_pool::kAllocMemoryPool)) {
  //   return false;
  // }
  // return IsDisableGeKernel() || memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocMemoryPool);
}

// Use enhanced memory pool when enable debug, enable log, enable prof, dry run and so on.
bool AscendMemoryPool::UseEnhancedMemoryPool() {
  return false;
}
} // namespace ascend
} // namespace device
} // namespace fxrt
