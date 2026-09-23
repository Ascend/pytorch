#ifndef FXRT_SRC_HARDWARE_MEMORY_ABSTRACT_DYNAMIC_MEM_POOL_H_
#define FXRT_SRC_HARDWARE_MEMORY_ABSTRACT_DYNAMIC_MEM_POOL_H_

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hardware/hardware_abstract/memory/dynamic_mem_pool.h"
#include "common/visible.h"
#include "hardware/hardware_abstract/stream_util.h"

namespace fxrt {
namespace device {
constexpr size_t kDecimalPrecision = 3;
// largest allocation size for small pool is 1 MB
constexpr size_t kSmallSize = 1048576;

struct FXRT_EXPORT MemBlock;

using MemBufStatus = DynamicMemBufStatus;
struct FXRT_EXPORT MemBuf : EventBase {
  explicit MemBuf(size_t size, void* addr, uint32_t streamId, MemBlock* memBlock, MemBufStatus status);

  MemBuf() = delete;
  MemBuf(const MemBuf&) = delete;
  MemBuf& operator=(const MemBuf&) = delete;

  ~MemBuf();

  inline void Link(MemBuf* prev, MemBuf* next) {
    if (prev != nullptr) {
      prev->next_ = this;
      this->prev_ = prev;
    }
    if (next != nullptr) {
      next->prev_ = this;
      this->next_ = next;
    }
  }

  inline void Unlink() {
    if (prev_ != nullptr) {
      prev_->next_ = next_;
    }
    if (next_ != nullptr) {
      next_->prev_ = prev_;
    }
    prev_ = nullptr;
    next_ = nullptr;
  }

  inline void SetDebugInfo() {
    ownerName_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
    allocType_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().type_;
  }

  std::string ToJson() {
    JsonBuilder builder;
    builder.Append("addr_", addr_);
    builder.Append("size_", size_);
    builder.Append("streamId_", streamId_);
    builder.Append("status_", DynamicMemBufStatusToString(status_));
    builder.Append("ownerName_", ownerName_);
    return builder.ToString();
  }

  MemBuf* prev_;
  MemBuf* next_;

  size_t size_;
  void* addr_;
  uint32_t streamId_;
  MemBlock* memBlock_;
  MemBufStatus status_;
  memory::mem_pool::MemType allocType_{memory::mem_pool::MemType::kOther};
  std::string ownerName_;
};

struct MemBufComparator {
  bool operator()(MemBuf* const& left, MemBuf* const& right) const {
    return (left->size_ != right->size_) ? left->size_ < right->size_ : left->addr_ < right->addr_;
  }
};

struct FXRT_EXPORT MemBlock {
  explicit MemBlock(size_t size, void* addr, uint32_t streamId) : size_(size), addr_(addr), streamId_(streamId) {
    minAddr_ = nullptr;
    maxAddr_ = nullptr;
  }

  MemBlock() = delete;
  MemBlock(const MemBlock&) = delete;
  MemBlock& operator=(const MemBlock&) = delete;

  ~MemBlock() = default;

  inline void UpdateBorderAddr(MemBuf* memBuf) {
    if (minAddr_ == nullptr) {
      minAddr_ = memBuf->addr_;
    } else {
      minAddr_ = std::min(minAddr_, memBuf->addr_);
    }
    void* right_addr = static_cast<uint8_t*>(memBuf->addr_) + memBuf->size_;
    maxAddr_ = std::max(maxAddr_, right_addr);
  }

  inline size_t ActualPeakSize() {
    if (minAddr_ == nullptr || maxAddr_ == nullptr) {
      return 0;
    }
    return static_cast<uint8_t*>(maxAddr_) - static_cast<uint8_t*>(minAddr_);
  }

  std::string ToJson() {
    JsonBuilder builder;
    builder.Append("addr_", addr_);
    builder.Append("size_", size_);
    builder.Append("streamId_", streamId_);
    builder.Append("minAddr_", minAddr_);
    builder.Append("maxAddr_", maxAddr_);
    return builder.ToString();
  }

  size_t size_;
  void* addr_;
  uint32_t streamId_;

  void* minAddr_;
  void* maxAddr_;
};

struct FXRT_EXPORT MemStat {
  MemStat() {
    Reset();
  }

  MemStat(const MemStat&) = delete;
  MemStat& operator=(const MemStat&) = delete;

  void Reset() {
    usedSize_ = 0;
    peakSize_ = 0;
    allocSize_ = 0;
    customAllocSize_ = 0;

    usedByEventSize_ = 0;
    eagerFreeSize_ = 0;

    iterUsedPeakSize_ = 0;
    iterAllocPeakSize_ = 0;
  }

  inline size_t IdleSize() const {
    return allocSize_ + customAllocSize_ - usedSize_;
  }

  inline void UpdatePeakSize(const bool isEnableVmm, size_t vmm_used_mem_size) {
    peakSize_ = std::max(peakSize_, usedSize_);
    iterUsedPeakSize_ = std::max(iterUsedPeakSize_, usedSize_);
    if (isEnableVmm) {
      iterAllocPeakSize_ = std::max(iterAllocPeakSize_, vmm_used_mem_size + customAllocSize_);
    } else {
      iterAllocPeakSize_ = std::max(iterAllocPeakSize_, allocSize_ + customAllocSize_);
    }
  }

  std::string ToJson() const {
    JsonBuilder builder;
    builder.Append("usedSize_", usedSize_);
    builder.Append("peakSize_", peakSize_);
    builder.Append("allocSize_", allocSize_);
    builder.Append("idle_size_", IdleSize());
    builder.Append("usedByEventSize_", usedByEventSize_);
    builder.Append("eagerFreeSize_", eagerFreeSize_);
    return builder.ToString();
  }

  std::string ToReadableString() const {
    JsonBuilder builder;
    builder.Append("in used mem", Format(usedSize_));
    builder.Append("peak used mem", Format(peakSize_));
    builder.Append("alloc mem", Format(allocSize_));
    builder.Append("idle mem", Format(IdleSize()));
    builder.Append("used by event mem", Format(usedByEventSize_));
    builder.Append("eager free mem", Format(eagerFreeSize_));
    return builder.ToString();
  }

  std::string Format(size_t size) const {
    auto str = std::to_string(size * 1.0 / kMBToByte);
    return str.substr(0, str.find(".") + kDecimalPrecision) + "MB";
  }

  size_t usedSize_;
  size_t peakSize_;
  size_t allocSize_;
  size_t customAllocSize_;

  size_t usedByEventSize_;
  size_t eagerFreeSize_;

  size_t iterUsedPeakSize_;
  size_t iterAllocPeakSize_;
};

struct AllocatorInfo {
  uint32_t streamId = 0;
  bool fromPersistentMem = false;
  bool use_small_pool = false;

  bool operator<(const AllocatorInfo& other) const {
    if (streamId != other.streamId) {
      return streamId < other.streamId;
    }
    if (fromPersistentMem != other.fromPersistentMem) {
      return other.fromPersistentMem;
    }
    if (use_small_pool != other.use_small_pool) {
      return other.use_small_pool;
    }
    return false;
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << "stream id: " << streamId << ", is persistent: " << fromPersistentMem
        << ", use small pool: " << use_small_pool;
    return oss.str();
  }
};

class AbstractDynamicMemPool;

class FXRT_EXPORT MemBufAllocator {
 public:
  explicit MemBufAllocator(
      std::function<MemBlock*(size_t)> memBlockExpander,
      std::function<bool(MemBlock*)> memBlockCleaner,
      std::function<size_t(size_t size, void* addr)> memMapper,
      std::function<size_t(void* addr, size_t size)> memEagerFreer,
      bool enableEagerFree,
      bool isPersistent,
      uint32_t streamId,
      bool isSmall,
      bool isCustomized = false)
      : memBlockExpander_(memBlockExpander),
        memBlockCleaner_(memBlockCleaner),
        memMapper_(memMapper),
        memEagerFreer_(memEagerFreer),
        streamId_(streamId),
        enableEagerFree_(enableEagerFree),
        isPersistent_(isPersistent),
        isSmall_(isSmall),
        isCustomized_(isCustomized) {
    searchKey_ = new MemBuf(0, nullptr, 0, nullptr, MemBufStatus::kMemBufIdle);
  }

  MemBufAllocator() = delete;
  MemBufAllocator(const MemBufAllocator&) = delete;
  MemBufAllocator& operator=(const MemBufAllocator&) = delete;

  ~MemBufAllocator();

  void Initialize(size_t size);
  void ReleaseDeviceRes();

  MemBuf* Malloc(size_t size);
  MemBuf* SearchAvailableMemBuf(size_t size);
  bool Free(MemBuf* memBuf, MemBufStatus targetStatus = MemBufStatus::kMemBufIdle);
  MemBuf* MallocExpandBlock(size_t size);
  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree();

  size_t ReleaseFreeBlocks();

  size_t ActualPeakSize() const {
    size_t peakSize = 0;
    for (auto memBlock : memBlocks_) {
      peakSize += memBlock->ActualPeakSize();
    }
    return peakSize;
  }

  std::string BriefInfo() const {
    std::stringstream ss;
    ss << "Mem buf allocator, enable vmm : " << enableEagerFree_ << ", is persistent : " << isPersistent_
       << ", stream id : " << streamId_ << ", is small : " << isSmall_ << ", is customized : " << isCustomized_ << ".";
    return ss.str();
  }

  uint32_t streamId() const {
    return streamId_;
  }
  bool isPersistent() const {
    return isPersistent_;
  }
  bool isSmall() const {
    return isSmall_;
  }
#ifndef ENABLE_TEST

 protected:
#endif
  MemBuf* MapAndSplitMemBuf(MemBuf* candidate, size_t size);
  MemBlock* ExpandBlock(size_t size);

  std::function<MemBlock*(size_t)> memBlockExpander_;
  std::function<bool(MemBlock*)> memBlockCleaner_;
  std::function<size_t(size_t size, void* addr)> memMapper_;
  std::function<size_t(void* addr, size_t size)> memEagerFreer_;

  std::list<MemBlock*> memBlocks_;
  using MemAllocator = memory::mem_pool::PooledAllocator<MemBuf*>;
  std::set<MemBuf*, MemBufComparator, MemAllocator> freeMemBufs_;
  std::set<MemBuf*, MemBufComparator, MemAllocator> eagerFreeMemBufs_;

 private:
  MemBuf* searchKey_;

  uint32_t streamId_;
  bool enableEagerFree_;
  bool isPersistent_;
  bool isSmall_;
  bool isCustomized_;

  friend AbstractDynamicMemPool;
};
using MemBufAllocatorPtr = std::shared_ptr<MemBufAllocator>;

using Lock = memory::mem_pool::Lock;
using LockGuard = memory::mem_pool::LockGuard;
class FXRT_EXPORT AbstractDynamicMemPool : virtual public DynamicMemPool {
 public:
  AbstractDynamicMemPool();
  ~AbstractDynamicMemPool() override = default;

  void Initialize(size_t initSize, size_t increaseSize, size_t maxSize) override;

  void ReleaseDeviceRes() override;

  // The main program entry of memory alloc.
  DeviceMemPtr AllocTensorMem(
      size_t size,
      bool fromPersistentMem = false,
      bool needRecycle = false,
      uint32_t streamId = kDefaultStreamIndex) override;

  // Alloc mem buf from mem pool, return mem buf and its allocator
  std::pair<MemBuf*, MemBufAllocator*> AllocMemBuf(
      size_t alignSize,
      bool fromPersistentMem = false,
      uint32_t streamId = kDefaultStreamIndex);

  // The main program entry of continuous memory alloc.
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(
      const std::vector<size_t>& sizeList,
      uint32_t streamId = kDefaultStreamIndex) override;
  // The main program entry of memory free.
  void FreeTensorMem(const DeviceMemPtr& deviceAddr) override;
  bool DoFreeTensorMem(const DeviceMemPtr& deviceAddr) override;
  // The main program entry of part memory free and part memory keep.
  void FreePartTensorMems(
      const std::vector<DeviceMemPtr>& freeAddrs,
      const std::vector<DeviceMemPtr>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes) override;
  virtual std::vector<MemBuf*> DoFreePartTensorMems(
      const std::vector<DeviceMemPtr>& freeAddrs,
      const std::vector<DeviceMemPtr>& keepAddrs,
      const std::vector<size_t>& keepAddrSizes);

  // Element in vector : memoryStreamId, address
  bool RecordEvent(
      int64_t taskIdOnStream,
      uint32_t userStreamId,
      const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
      const DeviceEventPtr& event) override;
  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) override;
  bool WaitEvent(int64_t taskIdOnStream, uint32_t memoryStreamId) override;
  bool SyncAllEvents() override;
  bool DoSyncAllEvents();

  size_t CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool needRecycle = false) override;
  void SetMemAllocUintSize(size_t commonSize, size_t persistSize = kDynamicMemAllocUnitSize) override {
    commonUnitSize_ = commonSize;
    persistUnitSize_ = persistSize;
  }
  size_t MemAllocUnitSize(bool fromPersistentMem = false) const override {
    return fromPersistentMem ? persistUnitSize_ : commonUnitSize_;
  }

  void DefragMemory() override;

  std::string DynamicMemPoolStateInfo() const;

  // The statistics information.
  size_t TotalMemStatistics() const override;
  size_t TotalUsedMemStatistics() const override;
  size_t TotalUsedByEventMemStatistics() const override;
  size_t TotalIdleMemStatistics() const override;
  size_t TotalEagerFreeMemStatistics() const override;
  size_t UsedMemPeakStatistics() const override;
  size_t MaxMemAllocatedStatistics() const override;
  size_t MaxMemReservedStatistics() const override;
  size_t ActualPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
      const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
      const override;
  void ResetMaxMemReserved() override;
  void ResetMaxMemAllocated() override;

  const bool IsEnableVmm() const override {
    return enableVmm_;
  }

  void SetEnableVmm(bool enableVmm) override {
    enableVmm_ = enableVmm;
  }

  // Get method for proxy.
  std::unordered_map<void*, std::pair<MemBuf*, MemBufAllocator*>>& addr_mem_buf_allocators() {
    return addrMemBufAllocators_;
  }

  std::unordered_map<std::pair<uint32_t, uint32_t>, std::set<MemBuf*>, pair_hash>& streamPairMemBufs() {
    return streamPairMemBufs_;
  }

  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() override;

  size_t ReleaseFreeBlocks() override;
  size_t ReleaseCustomFreeBlocks();

  MemStat& mem_stat() {
    return memStat_;
  }

  Lock& lock() {
    return lock_;
  }

 protected:
  void WaitPipelineHelper();

  MemBufAllocatorPtr GenerateAllocator(const AllocatorInfo& allocatorKey);
  MemBufAllocator* GetMemBufAllocator(size_t size, bool fromPersistentMem, uint32_t streamId);
#ifndef ENABLE_TEST

 protected:
#else

 public:
#endif
  std::map<AllocatorInfo, MemBufAllocatorPtr> streamIdAllocators_;
  std::unordered_map<void*, std::pair<MemBuf*, MemBufAllocator*>> addrMemBufAllocators_;
  std::unordered_map<std::pair<uint32_t, uint32_t>, std::set<MemBuf*>, pair_hash> streamPairMemBufs_;
  std::map<uint32_t, MemBufAllocatorPtr> customizedAllocators_;
  MemStat memStat_;

  bool enableVmm_{false};
  bool enableCustomAllocator_{false};
  std::function<MallocFuncType> customAllocFn_;
  std::function<FreeFuncType> customFreeFn_;
  size_t commonUnitSize_{kDynamicMemAllocUnitSize};
  size_t persistUnitSize_{kDynamicMemAllocUnitSize};

  size_t eagerFreeCount_{0};
  size_t lastEagerFreeCount_{0};
  Lock lock_;

  // initSize_ is for persistent and common.
  size_t initSize_{kDynamicMemAllocUnitSize};
  size_t increaseSize_{kDynamicMemAllocUnitSize};
  // Not enable currently.
  size_t maxSize_{0};

  bool enableDumpMemory_{false};
};

class FXRT_EXPORT AbstractEnhancedDynamicMemPool : public AbstractDynamicMemPool {
 public:
  AbstractEnhancedDynamicMemPool();
  AbstractEnhancedDynamicMemPool(const AbstractEnhancedDynamicMemPool&) = delete;
  AbstractEnhancedDynamicMemPool& operator=(const AbstractEnhancedDynamicMemPool&) = delete;
  ~AbstractEnhancedDynamicMemPool() override = default;

  // Report memory pool stat info for enhanced processing.
  virtual void ReportMemoryPoolInfo();
  // Report memory pool stat info for mstx
  virtual void ReportMemoryPoolMallocInfoToMstx(void* ptr, size_t size);
  virtual void ReportMemoryPoolFreeInfoToMstx(void* ptr);
  bool IsEnableTimeEvent() override {
    return enableTimeEvent_;
  }

  void SetEnableTimeEvent(bool enableTimeEvent) override {
    enableTimeEvent_ = enableTimeEvent;
  }

  virtual MemoryTimeEventPtr GenAllocateMemoryTimeEvent(
      const void* addr,
      size_t size,
      uint32_t streamId,
      bool fromPersistent,
      bool isPersistent);

  virtual MemoryTimeEventPtr GenFreeMemoryTimeEvent(const void* addr);

 private:
  std::atomic<bool> enableTimeEvent_{false};
};
} // namespace device
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_MEMORY_ABSTRACT_DYNAMIC_MEM_POOL_H_
