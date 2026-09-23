#include "hardware/hardware_abstract/memory/abstract_dynamic_mem_pool.h"

#include <stdio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <numeric>

#include "hardware/hardware_abstract/memory/mem_pool_util.h"
#include "common/logger.h"
#include "common/common.h"

namespace fxrt {
namespace device {
MemBuf::MemBuf(size_t size, void* addr, uint32_t streamId, MemBlock* memBlock, MemBufStatus status)
    : prev_(nullptr),
      next_(nullptr),
      size_(size),
      addr_(addr),
      streamId_(streamId),
      memBlock_(memBlock),
      status_(status) {}

MemBuf::~MemBuf() {}

MemBufAllocator::~MemBufAllocator() {
  RT_VLOG(VL_HARDWARE) << "MemBufAllocator[" << this << "] : " << BriefInfo() << " deconstruct.";
  for (auto& memBlock : memBlocks_) {
    delete memBlock;
  }
  memBlocks_.clear();
  for (auto memBuf : freeMemBufs_) {
    delete memBuf;
  }
  freeMemBufs_.clear();
  for (auto memBuf : eagerFreeMemBufs_) {
    delete memBuf;
  }
  eagerFreeMemBufs_.clear();
  delete searchKey_;
}

void MemBufAllocator::ReleaseDeviceRes() {
  RT_VLOG(VL_HARDWARE) << "Release device resource for allocator, " << BriefInfo()
                       << ", memBlocks_ size : " << memBlocks_.size() << ".";
  for (auto memBlock : memBlocks_) {
    RT_VLOG(VL_HARDWARE) << "Clean mem block : " << memBlock->ToJson() << ".";
    (void)memBlockCleaner_(memBlock);
  }
  for (auto memBlock : memBlocks_) {
    RT_VLOG(VL_HARDWARE) << "Delete mem block : " << memBlock->ToJson() << ".";
    delete memBlock;
  }
  memBlocks_.clear();

  RT_VLOG(VL_HARDWARE) << "Free mem buf size : " << freeMemBufs_.size() << ".";
  for (auto memBuf : freeMemBufs_) {
    delete memBuf;
  }
  freeMemBufs_.clear();

  RT_VLOG(VL_HARDWARE) << "Eager free mem buf size : " << eagerFreeMemBufs_.size() << ".";
  for (auto memBuf : eagerFreeMemBufs_) {
    delete memBuf;
  }
  eagerFreeMemBufs_.clear();
}

MemBuf* MemBufAllocator::Malloc(size_t size) {
  // Malloc with expand block first.
  if (FXRT_UNLIKELY(memBlocks_.empty())) {
    return MallocExpandBlock(size);
  }

  searchKey_->size_ = size;
  auto it = freeMemBufs_.lower_bound(searchKey_);
  MemBuf* candidate = nullptr;
  // 1. Try to find in free mem bufs.
  if (FXRT_LIKELY(it != freeMemBufs_.end())) {
    candidate = *it;
    (void)freeMemBufs_.erase(it);
    return MapAndSplitMemBuf(candidate, size);
  }
  // 2. Try to search available buf, free and eager free buf.
  candidate = SearchAvailableMemBuf(size);
  if (FXRT_UNLIKELY(candidate != nullptr)) {
    return candidate;
  }
  // 3. Try to find in eager free mem bufs.
  it = eagerFreeMemBufs_.lower_bound(searchKey_);
  if (it != eagerFreeMemBufs_.end()) {
    candidate = *it;
    (void)eagerFreeMemBufs_.erase(it);
    return MapAndSplitMemBuf(candidate, size);
  }

  return nullptr;
}

MemBuf* MemBufAllocator::SearchAvailableMemBuf(size_t size) {
  if (!enableEagerFree_ || FXRT_UNLIKELY(isCustomized_)) {
    return nullptr;
  }
  // Search from back to front, because the free mem buf is sorted by size.
  // More efficient way is to search more candidates, do it in the next version.
  for (auto backwardIt = freeMemBufs_.rbegin(); backwardIt != freeMemBufs_.rend(); backwardIt++) {
    auto memBuf = *backwardIt;
    auto nextBuf = memBuf->next_;
    if (nextBuf == nullptr || nextBuf->status_ != MemBufStatus::kMemBufEagerFree ||
        memBuf->size_ + nextBuf->size_ < size) {
      continue;
    }

    // Located candidates, try map and split.
    auto needMapSize = size - memBuf->size_;
    auto mappedSize = memMapper_(needMapSize, nextBuf->addr_);
    if (mappedSize != needMapSize) {
      RT_VLOG(VL_HARDWARE) << "Map mem buf : " << memBuf->ToJson() << ", next buf : " << nextBuf->ToJson()
                           << ", size : " << size << ", needMapSize : " << needMapSize
                           << ", mappedSize : " << mappedSize << " failed.";
      return nullptr;
    }
    // Update mem buf.
    freeMemBufs_.erase(memBuf);
    memBuf->size_ = size;
    memBuf->status_ = MemBufStatus::kMemBufUsed;
    // Remove eager free buf and try update it.
    eagerFreeMemBufs_.erase(nextBuf);
    nextBuf->addr_ = static_cast<uint8_t*>(nextBuf->addr_) + needMapSize;
    nextBuf->size_ = nextBuf->size_ - needMapSize;
    // If next buf is empty, remove it or update remain eager free mem buf.
    if (nextBuf->size_ == 0) {
      memBuf->next_ = nextBuf->next_;
      if (nextBuf->next_ != nullptr) {
        nextBuf->next_->prev_ = memBuf;
      }
      delete nextBuf;
    } else {
      eagerFreeMemBufs_.insert(nextBuf);
    }
    return memBuf;
  }
  return nullptr;
}

bool MemBufAllocator::Free(MemBuf* memBuf, MemBufStatus targetStatus) {
  // Change mem buf status to used by event, and wait for event to free.
  if (FXRT_UNLIKELY(!memBuf->IsEventNotUsed())) {
    memBuf->status_ = MemBufStatus::kMemBufUsedByEvent;
    return false;
  }

  memBuf->status_ = targetStatus;
  // Try to merge from prev.
  auto prevBuf = memBuf->prev_;
  if (FXRT_LIKELY(prevBuf != nullptr && prevBuf->status_ == targetStatus)) {
    // Erase prev buf pointer
    auto prev = prevBuf->prev_;
    memBuf->prev_ = prev;
    if (prev != nullptr) {
      prev->next_ = memBuf;
    }

    memBuf->addr_ = prevBuf->addr_;
    memBuf->size_ += prevBuf->size_;
    if (targetStatus == MemBufStatus::kMemBufIdle) {
      auto ret = freeMemBufs_.erase(prevBuf);
      if (ret == 0) {
        RT_GLOG(ERROR) << "Erase mem buf : " << memBuf->ToJson() << " prev buf " << prevBuf->ToJson() << " failed.";
      }
    } else if (targetStatus == MemBufStatus::kMemBufEagerFree) {
      auto ret = eagerFreeMemBufs_.erase(prevBuf);
      if (ret == 0) {
        RT_GLOG(ERROR) << "Erase mem buf : " << memBuf->ToJson() << " prev buf " << prevBuf->ToJson() << " failed.";
      }
    }
    delete prevBuf;
  }
  // Try to merge from next.
  auto nextBuf = memBuf->next_;
  if (FXRT_LIKELY(nextBuf != nullptr && nextBuf->status_ == targetStatus)) {
    // Erase next buf pointer
    auto next = nextBuf->next_;
    memBuf->next_ = next;
    if (next != nullptr) {
      next->prev_ = memBuf;
    }

    memBuf->size_ += nextBuf->size_;
    if (targetStatus == MemBufStatus::kMemBufIdle) {
      auto ret = freeMemBufs_.erase(nextBuf);
      if (ret == 0) {
        RT_GLOG(ERROR) << "Erase next buf : " << nextBuf->ToJson() << " failed.";
      }
    } else if (targetStatus == MemBufStatus::kMemBufEagerFree) {
      auto ret = eagerFreeMemBufs_.erase(nextBuf);
      if (ret == 0) {
        RT_GLOG(ERROR) << "Erase next buf : " << nextBuf->ToJson() << " failed.";
      }
    }
    delete nextBuf;
  }

  if (targetStatus == MemBufStatus::kMemBufIdle) {
    (void)freeMemBufs_.emplace(memBuf);
  } else if (targetStatus == MemBufStatus::kMemBufEagerFree) {
    (void)eagerFreeMemBufs_.emplace(memBuf);
  }

  return true;
}

MemBuf* MemBufAllocator::MallocExpandBlock(size_t size) {
  MemBlock* memBlock = ExpandBlock(size);
  if (memBlock == nullptr) {
    return nullptr;
  }
  MemBuf* candidate = new MemBuf(
      memBlock->size_,
      memBlock->addr_,
      memBlock->streamId_,
      memBlock,
      FXRT_LIKELY(!isCustomized_) && enableEagerFree_ ? MemBufStatus::kMemBufEagerFree : MemBufStatus::kMemBufIdle);
  if (candidate->size_ < size) {
    if (candidate->status_ == MemBufStatus::kMemBufIdle) {
      (void)freeMemBufs_.emplace(candidate);
    } else {
      (void)eagerFreeMemBufs_.emplace(candidate);
    }
    RT_VLOG(VL_HARDWARE) << "Candidate size: " << candidate->size_ << " is less than required size : " << size << ".";
    return nullptr;
  }

  return MapAndSplitMemBuf(candidate, size);
}

void MemBufAllocator::Initialize(size_t size) {
  RT_VLOG(VL_HARDWARE) << "Initialize allocator : " << BriefInfo() << " with size : " << size << ".";
  if (enableEagerFree_ || FXRT_UNLIKELY(isCustomized_)) {
    RT_VLOG(VL_HARDWARE) << "Skip initialization of allocator, since vmm is enabled.";
    return;
  }
  MemBlock* memBlock = ExpandBlock(size);
  if (memBlock == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Initialize allocator failed, size : " << size << ".";
    return;
  }
  MemBuf* memBuf =
      new MemBuf(memBlock->size_, memBlock->addr_, memBlock->streamId_, memBlock, MemBufStatus::kMemBufIdle);
  (void)freeMemBufs_.emplace(memBuf);
}

const std::pair<size_t, size_t> MemBufAllocator::FreeIdleMemsByEagerFree() {
  // Free all idle mem bufs.
  size_t eagerFreeSize = 0;
  for (auto memBuf : freeMemBufs_) {
    eagerFreeSize += memBuf->size_;
    Free(memBuf, MemBufStatus::kMemBufEagerFree);
  }
  freeMemBufs_.clear();
  // Do eager free on eager free mem bufs.
  size_t realFreeSize = 0;
  for (auto memBuf : eagerFreeMemBufs_) {
    RT_VLOG(VL_HARDWARE) << "Eager free mem buf : " << memBuf << ", details : " << memBuf->ToJson() << ".";
    realFreeSize += memEagerFreer_(memBuf->addr_, memBuf->size_);
  }
  RT_VLOG(VL_HARDWARE) << "Free idle mems by eager free, eagerFreeSize : " << eagerFreeSize
                       << ", realFreeSize : " << realFreeSize << ".";
  return std::make_pair(eagerFreeSize, realFreeSize);
}

size_t MemBufAllocator::ReleaseFreeBlocks() {
  size_t releaseSize = 0;
  for (auto iter = memBlocks_.begin(); iter != memBlocks_.end();) {
    auto memBlock = *iter;
    MemBuf memBuf(memBlock->size_, memBlock->addr_, memBlock->streamId_, memBlock, MemBufStatus::kMemBufIdle);
    // Judge if mem block in free mem bufs.
    auto&& it = freeMemBufs_.find(&memBuf);
    if (it == freeMemBufs_.end()) {
      iter++;
      continue;
    }
    auto memBufIt = *it;
    if (memBufIt->addr_ == memBlock->addr_ && memBufIt->size_ == memBlock->size_) {
      RT_VLOG(VL_HARDWARE) << "Release mem block : " << memBlock->ToJson() << ".";
      bool ret = memBlockCleaner_(memBlock);
      if (!ret) {
        RT_VLOG(VL_HARDWARE) << "Clean mem block : " << memBlock->ToJson() << " failed.";
        iter++;
        continue;
      }
      freeMemBufs_.erase(it);
      delete memBufIt;
      releaseSize += memBlock->size_;
      delete memBlock;
      iter = memBlocks_.erase(iter);
    } else {
      iter++;
    }
  }
  return releaseSize;
}

MemBuf* MemBufAllocator::MapAndSplitMemBuf(MemBuf* candidate, size_t size) {
  size_t remainingSize = candidate->size_ - size;
  // Mmap memory first.
  if (candidate->status_ == MemBufStatus::kMemBufEagerFree) {
    size_t mapSize = (remainingSize >= kDynamicMemAlignSize) ? size : candidate->size_;
    auto mappedSize = memMapper_(mapSize, candidate->addr_);
    if (mappedSize != mapSize) {
      RT_VLOG(VL_HARDWARE) << "Mapped_size : " << mappedSize << " is not equal to required size : " << mapSize
                           << ", mem buf info : " << candidate->ToJson() << ".";
      (void)eagerFreeMemBufs_.emplace(candidate);
      return nullptr;
    }
  }

  bool needSplit = remainingSize >= kDynamicMemAlignSize;
  // Try to split mem buf.
  if (FXRT_LIKELY(needSplit)) {
    void* remaining_addr = static_cast<uint8_t*>(candidate->addr_) + size;
    auto remainingBuf =
        new MemBuf(remainingSize, remaining_addr, candidate->streamId_, candidate->memBlock_, candidate->status_);

    auto next = candidate->next_;
    if (next != nullptr) {
      next->prev_ = remainingBuf;
      remainingBuf->next_ = next;
    }
    candidate->next_ = remainingBuf;
    remainingBuf->prev_ = candidate;
    if (remainingBuf->status_ == MemBufStatus::kMemBufIdle) {
      (void)freeMemBufs_.emplace(remainingBuf);
    } else {
      (void)eagerFreeMemBufs_.emplace(remainingBuf);
    }

    // Update candidate size.
    candidate->size_ = size;
  }

  candidate->status_ = MemBufStatus::kMemBufUsed;
  // Update mem block usage.
  candidate->memBlock_->UpdateBorderAddr(candidate);

  return candidate;
}

MemBlock* MemBufAllocator::ExpandBlock(size_t size) {
  MemBlock* memBlock = memBlockExpander_(size);
  if (memBlock == nullptr) {
    RT_VLOG(VL_HARDWARE) << "Expand block failed, expand size : " << size << ", memory is not enough.";
    return nullptr;
  }

  if (memBlock->size_ < size) {
    RT_VLOG(VL_HARDWARE) << "Expand block failed, expand size : " << memBlock->size_
                         << " is less than require size : " << size << ".";
  }

  (void)memBlocks_.emplace_back(memBlock);
  return memBlock;
}

AbstractDynamicMemPool::AbstractDynamicMemPool() {}

void AbstractDynamicMemPool::Initialize(size_t initSize, size_t increaseSize, size_t maxSize) {
  if (initSize == 0) {
    RT_VLOG(VL_HARDWARE) << "Skip initialization of memory pool since init size is not configured.";
    return;
  }

  LockGuard lock(lock_);
  RT_VLOG(VL_HARDWARE) << "Initialize dynamic memory pool, init size : " << initSize
                       << ", increase size : " << increaseSize << ", max size : " << maxSize << ".";
  initSize_ = initSize >> 1;
  increaseSize_ = increaseSize;
  maxSize_ = maxSize;

  // Do initialization with init size.
  auto persistentAllocator = GetMemBufAllocator(initSize_, true, kDefaultStreamIndex);
  persistentAllocator->Initialize(AlignMemorySize(initSize_));
  auto commonAllocator = GetMemBufAllocator(initSize_, false, kDefaultStreamIndex);
  commonAllocator->Initialize(AlignMemorySize(initSize_));
}

void AbstractDynamicMemPool::ReleaseDeviceRes() {
  LockGuard lock(lock_);
  for (const auto& iter : streamPairMemBufs_) {
    auto size = iter.second.size();
    RT_VLOG(VL_HARDWARE) << "Event referred streamPairMemBufs_[" << iter.first.first << "-" << iter.first.second
                         << "], size : " << size << ".";
  }
  // Clear map of address to mem buf.
  for (const auto& iter : addrMemBufAllocators_) {
    auto memBuf = iter.second.first;
    delete memBuf;
  }
  addrMemBufAllocators_.clear();

  RT_VLOG(VL_HARDWARE) << "Release device resource for " << GetMemoryPoolType() << " : " << memStat_.ToReadableString()
                       << ".";
  for (const auto& streamIdAllocator : streamIdAllocators_) {
    const auto& allocator = streamIdAllocator.second;
    allocator->ReleaseDeviceRes();
  }
  for (const auto& customizedAllocator : customizedAllocators_) {
    const auto& allocator = customizedAllocator.second;
    allocator->ReleaseDeviceRes();
  }
  streamIdAllocators_.clear();
  customizedAllocators_.clear();
  streamPairMemBufs_.clear();
  memStat_.Reset();
}

/**
 * @brief Alloc tensor mem.
 * Allocation follow steps below:
 *    1 align size
 *    2 find from current allocator, if failed transfer to 3
 *    3 find from another allocator, if failed transfer to 4
 *    4 do eager free and find from current allocator again, if failed transfer to 5
 *    5 expand block
 */
DeviceMemPtr AbstractDynamicMemPool::AllocTensorMem(size_t size, bool fromPersistentMem, bool, uint32_t streamId) {
  size_t alignSize = AlignMemorySize(size);
  LockGuard lock(lock_);
  auto&& memBufAllocator = AllocMemBuf(alignSize, fromPersistentMem, streamId);
  if (FXRT_UNLIKELY(memBufAllocator.first == nullptr)) {
    return nullptr;
  }

  (void)addrMemBufAllocators_.emplace(memBufAllocator.first->addr_, memBufAllocator);
  return memBufAllocator.first->addr_;
}

/**
 * @brief Alloc mem buf.
 * Strategy when vmm is disable:
 *    Persistent memory:  First malloc form its own pool, if fails, try to malloc from common pool.
 *    Common memory:  First malloc from its own pool, if fails, it will try to expand the pool.
 *                    If the expansion fails, try to malloc from persistent pool.
 */
std::pair<MemBuf*, MemBufAllocator*> AbstractDynamicMemPool::AllocMemBuf(
    size_t alignSize,
    bool fromPersistentMem,
    uint32_t streamId) {
  auto allocator = GetMemBufAllocator(alignSize, fromPersistentMem, streamId);

  auto memBuf = allocator->Malloc(alignSize);
  if (FXRT_UNLIKELY(memBuf == nullptr)) {
    // Enable malloc from another allocator when fromPersistentMem is true and vmm is not enabled.
    if (!enableVmm_ && fromPersistentMem && FXRT_LIKELY(!enableCustomAllocator_)) {
      auto commonAllocator = GetMemBufAllocator(alignSize, false, streamId);
      memBuf = commonAllocator->Malloc(alignSize);
      allocator = commonAllocator;
    }

    if (FXRT_UNLIKELY(memBuf == nullptr)) {
      if ((enableVmm_ || IsEnableEagerFree()) && FXRT_LIKELY(!enableCustomAllocator_)) {
        WaitPipelineHelper();
        if (!SyncAllStreams()) {
          RT_GLOG(ERROR) << "Sync all streams failed.";
          return std::make_pair(nullptr, nullptr);
        }
        (void)FreeIdleMemsByEagerFree();
        memBuf = allocator->Malloc(alignSize);
      }
      if (FXRT_UNLIKELY(memBuf == nullptr)) {
        memBuf = allocator->MallocExpandBlock(alignSize);
        if (FXRT_UNLIKELY(memBuf == nullptr)) {
          if (FXRT_LIKELY(!fromPersistentMem) && FXRT_LIKELY(!enableCustomAllocator_)) {
            // Common pool expand block failed, try to malloc from persistent pool.
            auto persistentAllocator = GetMemBufAllocator(alignSize, true, streamId);
            memBuf = persistentAllocator->Malloc(alignSize);
            if (FXRT_LIKELY(memBuf != nullptr)) {
              allocator = persistentAllocator;
            }
          }

          if (FXRT_UNLIKELY(memBuf == nullptr)) {
            RT_VLOG(VL_HARDWARE) << "Alloc tensor mem failed and try to sync all events to release memory.";
            (void)DoSyncAllEvents();
            memBuf = allocator->Malloc(alignSize);
            if (FXRT_UNLIKELY(memBuf == nullptr)) {
              return std::make_pair(nullptr, nullptr);
            }
          }
        }
      }
    }
  }

  // Update stat.
  memStat_.usedSize_ += memBuf->size_;
  memStat_.UpdatePeakSize(enableVmm_, GetVmmUsedMemSize());
  return std::make_pair(memBuf, allocator);
}

std::vector<DeviceMemPtr> AbstractDynamicMemPool::AllocContinuousTensorMem(
    const std::vector<size_t>& sizeList,
    uint32_t streamId) {
  std::vector<DeviceMemPtr> deviceAddrList;
  size_t totalSize = std::accumulate(sizeList.begin(), sizeList.end(), static_cast<size_t>(0));
  // Pre-alloc the one whole piece memory.
  auto deviceAddr = AbstractDynamicMemPool::AllocTensorMem(totalSize, false, false, streamId);
  if (deviceAddr == nullptr) {
    return deviceAddrList;
  }

  (void)deviceAddrList.emplace_back(deviceAddr);
  if (sizeList.size() == 1) {
    return deviceAddrList;
  }

  // Try to split mem bufs.
  LockGuard lock(lock_);
  auto&& it = addrMemBufAllocators_.find(deviceAddr);
  if (it != addrMemBufAllocators_.end()) {
    auto memBuf = it->second.first;
    auto allocator = it->second.second;
    memBuf->size_ = sizeList[0];
    MemBuf* prevMemBuf = memBuf;
    void* nextAddr = static_cast<uint8_t*>(memBuf->addr_) + sizeList[0];
    totalSize -= sizeList[0];
    for (size_t i = 1; i < sizeList.size(); i++) {
      auto newMemBuf = new MemBuf(sizeList[i], nextAddr, streamId, memBuf->memBlock_, MemBufStatus::kMemBufUsed);
      newMemBuf->Link(prevMemBuf, prevMemBuf->next_);
      (void)addrMemBufAllocators_.emplace(newMemBuf->addr_, std::make_pair(newMemBuf, allocator));
      // Update result.
      (void)deviceAddrList.emplace_back(nextAddr);
      // Update next addr and prev mem buf.
      if (i < sizeList.size() - 1) {
        nextAddr = static_cast<uint8_t*>(nextAddr) + sizeList[i];
        totalSize -= sizeList[i];
        prevMemBuf = newMemBuf;
      } else {
        // Update last mem buf
        if (totalSize != sizeList[i]) {
          RT_VLOG(VL_HARDWARE) << "Remain size : " << totalSize << " is not equal to last size : " << sizeList[i]
                               << ".";
          newMemBuf->size_ = totalSize;
        }
      }
    }
  } else {
    // Unreachable routine.
    RT_GLOG(ERROR) << "Find addr : " << deviceAddr << " failed.";
  }

  return deviceAddrList;
}

// The main program entry of memory free.
void AbstractDynamicMemPool::FreeTensorMem(const DeviceMemPtr& deviceAddr) {
  LockGuard lock(lock_);
  (void)DoFreeTensorMem(deviceAddr);
}

// The main program entry of memory free.
bool AbstractDynamicMemPool::DoFreeTensorMem(const DeviceMemPtr& deviceAddr) {
  void* addr = deviceAddr;
  auto&& it = addrMemBufAllocators_.find(deviceAddr);
  if (FXRT_LIKELY(it != addrMemBufAllocators_.end())) {
    auto allocator = it->second.second;
    auto memBuf = it->second.first;
    auto freeSize = memBuf->size_;
    if (FXRT_LIKELY(allocator->Free(memBuf))) {
      memStat_.usedSize_ -= freeSize;
      (void)addrMemBufAllocators_.erase(it);
      return true;
    }
  } else {
    // This may be normal case.
    RT_VLOG(VL_HARDWARE) << "Free tensor mem failed, can not find address : " << addr << ".";
  }
  return false;
}

MemBufAllocator* AbstractDynamicMemPool::GetMemBufAllocator(size_t size, bool fromPersistentMem, uint32_t streamId) {
  // Not use small pool.
  const AllocatorInfo key{streamId, fromPersistentMem, false};
  RT_VLOG(VL_HARDWARE) << "Get allocator, " << key.ToString() << ".";

  MemBufAllocatorPtr allocator = nullptr;

  auto&& it = streamIdAllocators_.find(key);
  if (it == streamIdAllocators_.end()) {
    allocator = GenerateAllocator(key);
    (void)streamIdAllocators_.emplace(key, allocator);
  } else {
    allocator = it->second;
  }
  return allocator.get();
}

// Keep addrs is in free addrs, so here find mem bufs first.
// And then, traverse keep addrs and spilt candidates.
void AbstractDynamicMemPool::FreePartTensorMems(
    const std::vector<DeviceMemPtr>& freeAddrs,
    const std::vector<DeviceMemPtr>& keepAddrs,
    const std::vector<size_t>& keepAddrSizes) {
  RT_VLOG(VL_HARDWARE) << "Free part tensor mems.";
  LockGuard lock(lock_);
  (void)DoFreePartTensorMems(freeAddrs, keepAddrs, keepAddrSizes);
}

std::vector<MemBuf*> AbstractDynamicMemPool::DoFreePartTensorMems(
    const std::vector<DeviceMemPtr>& freeAddrs,
    const std::vector<DeviceMemPtr>& keepAddrs,
    const std::vector<size_t>& keepAddrSizes) {
  std::vector<MemBuf*> memBufs;
  std::map<void*, std::pair<MemBuf*, MemBufAllocator*>> candidates;
  for (const auto& free_addr : freeAddrs) {
    auto&& it = addrMemBufAllocators_.find(free_addr);
    if (it != addrMemBufAllocators_.end()) {
      (void)candidates.emplace(it->first, it->second);
    } else {
      // This is illegal routine, but level0 case entered.
      RT_VLOG(VL_HARDWARE) << "Find address : " << free_addr << " failed.";
    }
  }

  std::set<std::uintptr_t> processedKeepAddrs;
  for (size_t i = 0; i < keepAddrs.size(); i++) {
    auto keepAddr = keepAddrs[i];
    std::uintptr_t keepAddrToSize = reinterpret_cast<std::uintptr_t>(keepAddr);
    if (processedKeepAddrs.count(keepAddrToSize) > 0) {
      RT_VLOG(VL_HARDWARE) << "Duplicate keep address : " << keepAddr << ".";
      continue;
    }
    (void)processedKeepAddrs.insert(keepAddrToSize);
    auto&& it = candidates.upper_bound(keepAddr);
    if (it == candidates.begin()) {
      RT_VLOG(VL_HARDWARE) << "Locate keep addr : " << keepAddr << " failed.";
      continue;
    }
    auto iter = --it;
    auto memBuf = iter->second.first;
    auto allocator = iter->second.second;
    std::uintptr_t baseStart = reinterpret_cast<std::uintptr_t>(memBuf->addr_);
    std::uintptr_t baseEnd = baseStart + memBuf->size_;
    std::uintptr_t keepStart = keepAddrToSize;
    std::uintptr_t keepEnd = keepStart + keepAddrSizes[i];
    // Since free part tensor mem may double free keep addr, continue for these keep addrs.
    if (keepStart >= baseEnd) {
      RT_VLOG(VL_HARDWARE) << "Check range error, base start : " << baseStart << ", base end : " << baseEnd
                           << ", keep start : " << keepStart << ", keep end : " << keepEnd << ".";
      continue;
    }
    // Split candidates. If keep start equal to base start, split mem buf into two parts, or three parts.
    // First construct keep mem buf and set it into addrMemBufAllocators_, then process head buf and tail buf.
    MemBuf* keepMemBuf = nullptr;
    if (keepStart == baseStart) {
      keepMemBuf = memBuf;
      keepMemBuf->size_ = keepAddrSizes[i];
      // Remove keep addr since keep start equal to base start, no need to free keep addr any more.
      (void)candidates.erase(memBuf->addr_);
    } else {
      // Split middle mem buf.
      keepMemBuf = new MemBuf(keepAddrSizes[i], keepAddr, memBuf->streamId_, memBuf->memBlock_, memBuf->status_);
      keepMemBuf->Link(memBuf, memBuf->next_);
      (void)addrMemBufAllocators_.emplace(keepAddr, std::make_pair(keepMemBuf, allocator));
      std::uintptr_t prevRemainSize = keepStart - baseStart;
      memBuf->size_ = prevRemainSize;
    }
    (void)memBufs.emplace_back(keepMemBuf);
    RT_VLOG(VL_HARDWARE) << "keepMemBuf : " << keepMemBuf->ToJson() << ".";
    // Process last mem buf.
    if (keepEnd < baseEnd) {
      void* lastAddr = static_cast<uint8_t*>(keepMemBuf->addr_) + keepMemBuf->size_;
      auto lastMemBuf =
          new MemBuf(baseEnd - keepEnd, lastAddr, keepMemBuf->streamId_, keepMemBuf->memBlock_, memBuf->status_);
      lastMemBuf->Link(keepMemBuf, keepMemBuf->next_);
      (void)addrMemBufAllocators_.emplace(lastMemBuf->addr_, std::make_pair(lastMemBuf, allocator));
      if (candidates.count(lastMemBuf->addr_) > 0) {
        RT_VLOG(VL_HARDWARE) << "Duplicate address : " << lastMemBuf->addr_ << ".";
      }
      RT_VLOG(VL_HARDWARE) << "last mem buf : " << lastMemBuf->ToJson() << ".";
      (void)candidates.emplace(lastMemBuf->addr_, std::make_pair(lastMemBuf, allocator));
    }
  }
  for (const auto& candidate : candidates) {
    auto memBuf = candidate.second.first;
    if (!AbstractDynamicMemPool::DoFreeTensorMem(memBuf->addr_)) {
      RT_GLOG(ERROR) << "Free device address failed : " << memBuf->addr_ << ", memBuf : " << memBuf->ToJson() << ".";
    }
  }
  return memBufs;
}

MemBufAllocatorPtr AbstractDynamicMemPool::GenerateAllocator(const AllocatorInfo& allocatorKey) {
  const auto isPersistent = allocatorKey.fromPersistentMem;
  const auto streamId = allocatorKey.streamId;
  const auto isSmall = allocatorKey.use_small_pool;

  RT_VLOG(VL_HARDWARE) << "Generate allocator, " << allocatorKey.ToString() << ".";
  std::function<MemBlock*(size_t)> memBlockExpander =
      [&, isPersistent = isPersistent, streamId = streamId](size_t size) {
        size_t blockSize = CalMemBlockAllocSize(size, isPersistent);
        MemBlock* memBlock = nullptr;
        if (blockSize == 0) {
          RT_VLOG(VL_HARDWARE) << "Malloc mem block failed, is enable eager free : " << IsEnableEagerFree()
                               << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ", block size is  0.";
          return memBlock;
        }
        DeviceMemPtr addr = nullptr;
        size_t allocSize;
        RT_VLOG(VL_HARDWARE) << "Malloc mem block, is enable eager free : " << IsEnableEagerFree()
                             << ", is enable vmm : " << IsEnableVmm() << ", size : " << size
                             << ", block size : " << blockSize << ".";
        if (IsEnableVmm() || IsEnableEagerFree()) {
          // Virtual address is unlimited.
          auto eagerFreeSize = std::max(blockSize, static_cast<size_t>(total_mem_size()));
          allocSize = AllocDeviceMemByEagerFree(eagerFreeSize, &addr);
          memStat_.eagerFreeSize_ += allocSize;
        } else {
          allocSize = AllocDeviceMem(blockSize, &addr);
          if (allocSize < blockSize) {
            RT_VLOG(VL_HARDWARE) << "Alloc device mem failed, alloc size : " << allocSize
                                 << ", block size : " << blockSize << ".";
          }
        }
        if (allocSize == 0) {
          return memBlock;
        }
        memStat_.allocSize_ += allocSize;
        memBlock = new MemBlock(allocSize, addr, streamId);
        RT_VLOG(VL_HARDWARE) << "Malloc mem block : " << memBlock->ToJson() << ".";
        return memBlock;
      };

  std::function<bool(MemBlock*)> memBlockCleaner = [&](MemBlock* memBlock) {
    memStat_.allocSize_ -= memBlock->size_;
    // Call free device mem as ascend memory pool would do stat in free operation.
    return FreeDeviceMem(memBlock->addr_);
  };
  std::function<size_t(size_t size, void* addr)> memMapper = [&](size_t size, void* addr) {
    memStat_.eagerFreeSize_ -= size;
    return MmapDeviceMem(size, addr);
  };
  std::function<size_t(void* addr, const size_t size)> memEagerFreer = [&](void* addr, const size_t size) {
    RT_VLOG(VL_HARDWARE) << "Eager free addr : " << addr << ", size : " << size << ".";
    return FreeDeviceMemByEagerFree(addr, size);
  };

  return std::make_shared<MemBufAllocator>(
      memBlockExpander,
      memBlockCleaner,
      memMapper,
      memEagerFreer,
      IsEnableVmm() || IsEnableEagerFree(),
      isPersistent,
      streamId,
      isSmall);
}

// Element in vector : <memoryStreamId, addr>
bool AbstractDynamicMemPool::RecordEvent(
    int64_t taskIdOnStream,
    uint32_t userStreamId,
    const std::vector<std::pair<uint32_t, DeviceMemPtr>>& memoryStreamAddresses,
    const DeviceEventPtr& event) {
  RT_VLOG(VL_HARDWARE) << "Record event for task id on stream : " << taskIdOnStream
                       << ", user stream id : " << userStreamId << ".";
  LockGuard lock(lock_);
  for (auto& [memoryStreamId, addr] : memoryStreamAddresses) {
    auto&& it = addrMemBufAllocators_.find(addr);
    if (it != addrMemBufAllocators_.end()) {
      auto memBuf = it->second.first;
      if (memBuf->IsEventNotUsed()) {
        memStat_.usedByEventSize_ += memBuf->size_;
      }
      RT_VLOG(VL_HARDWARE) << "Record event for : " << memBuf->ToJson() << ".";
      (void)memBuf->RecordEvent(taskIdOnStream, userStreamId, event);
      (void)streamPairMemBufs_[std::make_pair(userStreamId, memoryStreamId)].emplace(memBuf);
    } else {
      // Output of somas sub graph may be used by somas sub graph inner node, address may not be kept in mem pool.
      RT_VLOG(VL_HARDWARE) << "Unknown address : " << addr << ".";
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) {
  RT_VLOG(VL_HARDWARE) << "Wait event for task id on stream : " << taskIdOnStream
                       << ", user stream id : " << userStreamId << ", memory stream id : " << memoryStreamId << ".";
  LockGuard lock(lock_);
  auto key = std::make_pair(userStreamId, memoryStreamId);
  auto iter = streamPairMemBufs_.find(key);
  if (iter == streamPairMemBufs_.end()) {
    return false;
  }

  auto memBufs_ = iter->second;
  for (const auto& memBuf : memBufs_) {
    RT_VLOG(VL_HARDWARE) << "Wait event for : " << memBuf->ToJson() << ".";
    memBuf->WaitEvent(taskIdOnStream, userStreamId);
    // Remove event and try to free memory.
    if (memBuf->IsEventNotUsed()) {
      memStat_.usedByEventSize_ -= memBuf->size_;
      // Force clear all mem bufs.
      for (auto& streamPairMemBufs : streamPairMemBufs_) {
        (void)streamPairMemBufs.second.erase(memBuf);
      }
      if (memBuf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        (void)DoFreeTensorMem(memBuf->addr_);
      }
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t taskIdOnStream, uint32_t memoryStreamId) {
  RT_VLOG(VL_HARDWARE) << "Wait event for task id on stream : " << taskIdOnStream
                       << ", memory stream id : " << memoryStreamId << ".";
  LockGuard lock(lock_);
  for (auto& streamPairMemBufs : streamPairMemBufs_) {
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
        memStat_.usedByEventSize_ -= memBuf->size_;
        // Force clear all mem bufs.
        for (auto& kv : streamPairMemBufs_) {
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

bool AbstractDynamicMemPool::SyncAllEvents() {
  RT_VLOG(VL_HARDWARE) << "Sync all events, stream_pair_addresses_ size : " << streamPairMemBufs_.size() << ".";
  LockGuard lock(lock_);
  return DoSyncAllEvents();
}

bool AbstractDynamicMemPool::DoSyncAllEvents() {
  if (streamPairMemBufs_.empty()) {
    return false;
  }

  std::set<MemBuf*> carryEventMemBufs;
  for (const auto& streamPairMemBuf : streamPairMemBufs_) {
    for (const auto& memBuf : streamPairMemBuf.second) {
      (void)carryEventMemBufs.emplace(memBuf);
    }
  }
  for (auto& memBuf : carryEventMemBufs) {
    if (memBuf->SyncAllEvents() && memBuf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      (void)DoFreeTensorMem(memBuf->addr_);
    }
  }

  streamPairMemBufs_.clear();
  return true;
}

size_t AbstractDynamicMemPool::CalMemBlockAllocSize(size_t size, bool fromPersistentMem, bool) {
  auto deviceFreeMemSize = free_mem_size();
  // Make sure available mem is enough.
  if (deviceFreeMemSize < size) {
    RT_VLOG(VL_HARDWARE) << "Memory not enough: current free memory size[" << deviceFreeMemSize
                         << "] is smaller than required size[" << size << "].";
    return 0;
  }
  auto unitSize = MemAllocUnitSize(fromPersistentMem);
  if (unitSize == 0) {
    RT_GLOG(ERROR) << "Invalid memory alloc unit size 0, fall back to the default unit size ["
                   << kDynamicMemAllocUnitSize << "].";
    unitSize = kDynamicMemAllocUnitSize;
  }
  if (deviceFreeMemSize < unitSize) {
    RT_VLOG(VL_HARDWARE) << "Device memory size [" << deviceFreeMemSize << "] is smaller than unit size [" << unitSize
                         << "].";
  }
  // Calculate alloc size.
  size_t allocSize = unitSize;
  if (size > unitSize) {
    allocSize = ((size + unitSize - 1) / unitSize) * unitSize;
  }
  return std::min(allocSize, deviceFreeMemSize);
}

void AbstractDynamicMemPool::DefragMemory() {
  RT_VLOG(VL_HARDWARE) << "Try to defrag memory.";
  LockGuard lock(lock_);

  if (!enableVmm_) {
    RT_VLOG(VL_HARDWARE) << "Skip defrag memory since vmm is not enabled.";
    return;
  }

  if (eagerFreeCount_ == 0) {
    RT_VLOG(VL_HARDWARE) << "Exit defrag memory since eager free count is 0.";
    return;
  }
  if (lastEagerFreeCount_ == eagerFreeCount_) {
    RT_VLOG(VL_HARDWARE) << "Exit defrag memory since last eager free count equals to eager free count : "
                         << lastEagerFreeCount_ << ".";
    return;
  }

  RT_VLOG(VL_HARDWARE) << "Defrag memory start.";
  WaitPipelineHelper();
  if (!SyncAllStreams()) {
    RT_GLOG(ERROR) << "Sync all streams failed.";
    return;
  }
  const auto [eagerFreeSize, realFreeSize] = FreeIdleMemsByEagerFree();
  RT_VLOG(VL_HARDWARE) << "Defrag memory, eagerFreeSize : " << eagerFreeSize << ", realFreeSize : " << realFreeSize
                       << ".";
  lastEagerFreeCount_ = eagerFreeCount_;
}

void AbstractDynamicMemPool::WaitPipelineHelper() {
  if (pipelineCallback_) {
    lock_.unlock();
    pipelineCallback_();
    lock_.lock();
  }
}

std::string AbstractDynamicMemPool::DynamicMemPoolStateInfo() const {
  std::stringstream ss;
  // Classify mem buf and stat mem buf state info.
  size_t memBufUsedStat[static_cast<int>(memory::mem_pool::MemType::kOther) + 1] = {0};
  struct AddrComparator {
    bool operator()(MemBuf* const& left, MemBuf* const& right) const {
      return left->addr_ < right->addr_;
    }
  };
  std::map<MemBufAllocator*, std::set<MemBuf*, AddrComparator>> allocatorMemBufs;
  for (const auto& addrMemBufAllocator : addrMemBufAllocators_) {
    const auto allocator = addrMemBufAllocator.second.second;
    const auto memBuf = addrMemBufAllocator.second.first;
    memBufUsedStat[static_cast<int>(memBuf->allocType_)] += memBuf->size_;
    auto& memBufs = allocatorMemBufs[allocator];
    (void)memBufs.insert(memBuf);
  }
  for (const auto& [allocator, memBufs] : allocatorMemBufs) {
    ss << "\tIn used mem buf info for " << allocator->BriefInfo() << ", memBufs size : " << memBufs.size() << "\n";
  }

  size_t otherUsedSize = 0;
  int start = static_cast<int>(memory::mem_pool::MemType::kGraphOutput);
  int end = static_cast<int>(memory::mem_pool::MemType::kOther);
  for (int i = start; i <= end; i++) {
    otherUsedSize += memBufUsedStat[i];
  }

  ss << "The dynamic memory pool[" << GetMemoryPoolType() << "] stat info : " << memStat_.ToReadableString()
     << ", actual peak used mem:" << ActualPeakStatistics() / kMBToByte
     << "M. Weight used size:" << memBufUsedStat[static_cast<int>(memory::mem_pool::MemType::kWeight)] / kMBToByte
     << "M, constant value used size:"
     << memBufUsedStat[static_cast<int>(memory::mem_pool::MemType::kConstantValue)] / kMBToByte
     << "M, kernel output used size:"
     << memBufUsedStat[static_cast<int>(memory::mem_pool::MemType::kKernel)] / kMBToByte
     << "M, other used size:" << otherUsedSize / kMBToByte << "M.\n";
  return ss.str();
}

const std::pair<size_t, size_t> AbstractDynamicMemPool::FreeIdleMemsByEagerFree() {
  if (!IsEnableVmm() && !IsEnableEagerFree()) {
    RT_VLOG(VL_HARDWARE) << "FreeIdleMemsByEagerFree is not allowed since vmm is not enabled.";
    return std::make_pair(0L, 0L);
  }

  RT_VLOG(VL_HARDWARE) << "Free idle mems by eager free start, allocator size : " << streamIdAllocators_.size() << ".";
  eagerFreeCount_++;

  size_t totalEagerFreeSize = 0;
  size_t totalRealFreeSize = 0;
  for (auto& streamIdAllocator : streamIdAllocators_) {
    const auto [eagerFreeSize, realFreeSize] = streamIdAllocator.second->FreeIdleMemsByEagerFree();
    totalEagerFreeSize += eagerFreeSize;
    totalRealFreeSize += realFreeSize;
  }

  size_t notFreeSize = totalEagerFreeSize > totalRealFreeSize ? (totalEagerFreeSize - totalRealFreeSize) : 0;
  if (totalRealFreeSize >= kGBToByte) {
    RT_VLOG(VL_HARDWARE) << "Eager free count : " << eagerFreeCount_ << ", free memory : " << totalEagerFreeSize
                         << ", real free : " << totalRealFreeSize << ", not free : " << notFreeSize << ".";
  } else {
    RT_VLOG(VL_HARDWARE) << "Eager free count : " << eagerFreeCount_ << ", free memory : " << totalEagerFreeSize
                         << ", real free : " << totalRealFreeSize << ", not free : " << notFreeSize << ".";
  }

  memStat_.eagerFreeSize_ += totalEagerFreeSize;
  return {totalEagerFreeSize, totalRealFreeSize};
}

size_t AbstractDynamicMemPool::ReleaseFreeBlocks() {
  RT_VLOG(VL_HARDWARE) << "Release free blocks start.";
  size_t releaseSize = 0;
  for (auto& streamIdAllocator : streamIdAllocators_) {
    releaseSize += streamIdAllocator.second->ReleaseFreeBlocks();
  }
  RT_VLOG(VL_HARDWARE) << "Release free blocks size : " << releaseSize << ".";
  return releaseSize;
}

size_t AbstractDynamicMemPool::ReleaseCustomFreeBlocks() {
  RT_VLOG(VL_HARDWARE) << "Release custom free blocks start.";
  size_t releaseSize = 0;
  for (auto& customizedAllocator : customizedAllocators_) {
    releaseSize += customizedAllocator.second->ReleaseFreeBlocks();
  }
  RT_VLOG(VL_HARDWARE) << "Release custom free blocks size : " << releaseSize << ".";
  return releaseSize;
}

// The statistics information.
size_t AbstractDynamicMemPool::TotalMemStatistics() const {
  if (IsEnableVmm()) {
    return GetVmmUsedMemSize() + memStat_.customAllocSize_;
  }
  return memStat_.allocSize_ + memStat_.customAllocSize_;
}

size_t AbstractDynamicMemPool::TotalUsedMemStatistics() const {
  return memStat_.usedSize_;
}

size_t AbstractDynamicMemPool::TotalUsedByEventMemStatistics() const {
  return memStat_.usedByEventSize_;
}

size_t AbstractDynamicMemPool::TotalIdleMemStatistics() const {
  return memStat_.IdleSize();
}

size_t AbstractDynamicMemPool::TotalEagerFreeMemStatistics() const {
  return memStat_.eagerFreeSize_;
}

size_t AbstractDynamicMemPool::UsedMemPeakStatistics() const {
  return memStat_.peakSize_;
}

size_t AbstractDynamicMemPool::MaxMemAllocatedStatistics() const {
  return memStat_.iterUsedPeakSize_;
}

size_t AbstractDynamicMemPool::MaxMemReservedStatistics() const {
  return memStat_.iterAllocPeakSize_;
}

size_t AbstractDynamicMemPool::ActualPeakStatistics() const {
  if (IsEnableVmm()) {
    return GetVmmUsedMemSize() + memStat_.customAllocSize_;
  }

  size_t peakSize = 0;
  for (auto& streamIdAllocator : streamIdAllocators_) {
    peakSize += streamIdAllocator.second->ActualPeakSize();
  }
  for (auto& customizedAllocator : customizedAllocators_) {
    peakSize += customizedAllocator.second->ActualPeakSize();
  }
  return peakSize;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockCountsStatistics() const {
  LockGuard lock(lock_);
  size_t persistentBlockCount = 0;
  size_t commonBlockCount = 0;
  for (const auto& [allocatorInfo, allocatorPtr] : streamIdAllocators_) {
    if (allocatorInfo.fromPersistentMem) {
      persistentBlockCount += allocatorPtr->memBlocks_.size();
    } else {
      commonBlockCount += allocatorPtr->memBlocks_.size();
    }
  }
  std::unordered_map<std::string, size_t> blockCounts;
  blockCounts[kPersistentMemPoolType] = persistentBlockCount;
  blockCounts[kCommonMemPoolType] = commonBlockCount;
  return blockCounts;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockUnitSizeStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<std::string, size_t> blockUnits;
  blockUnits[kPersistentMemPoolType] = persistUnitSize_;
  blockUnits[kCommonMemPoolType] = commonUnitSize_;
  return blockUnits;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> AbstractDynamicMemPool::
    CommonMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> blockInfos;
  for (const auto& [allocatorInfo, allocatorPtr] : streamIdAllocators_) {
    if (!allocatorInfo.fromPersistentMem) {
      const auto& mem_blocks = allocatorPtr->memBlocks_;
      for (const auto memBlock : mem_blocks) {
        std::unordered_map<std::string, size_t> blockInfo;
        blockInfo[kBlockMemorySize] = memBlock->size_;
        blockInfo[kBlockStreamId] = memBlock->streamId_;
        blockInfos[(std::string*)(memBlock->addr_)] = blockInfo;
      }
    }
  }
  return blockInfos;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> AbstractDynamicMemPool::
    PersistentMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> blockInfos;
  for (const auto& [allocatorInfo, allocatorPtr] : streamIdAllocators_) {
    if (allocatorInfo.fromPersistentMem) {
      const auto& mem_blocks = allocatorPtr->memBlocks_;
      for (const auto memBlock : mem_blocks) {
        std::unordered_map<std::string, size_t> blockInfo;
        blockInfo[kBlockMemorySize] = memBlock->size_;
        blockInfo[kBlockStreamId] = memBlock->streamId_;
        blockInfos[(std::string*)(memBlock->addr_)] = blockInfo;
      }
    }
  }
  return blockInfos;
}

void AbstractDynamicMemPool::ResetMaxMemReserved() {
  LockGuard lock(lock_);
  memStat_.iterAllocPeakSize_ =
      IsEnableVmm() ? GetVmmUsedMemSize() + memStat_.customAllocSize_ : memStat_.allocSize_ + memStat_.customAllocSize_;
}

void AbstractDynamicMemPool::ResetMaxMemAllocated() {
  LockGuard lock(lock_);
  memStat_.iterUsedPeakSize_ = memStat_.usedSize_;
}

AbstractEnhancedDynamicMemPool::AbstractEnhancedDynamicMemPool() {}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolInfo() {
  // Report memory data to profiler.
  if (memoryProfilerCallback_) {
    memoryProfilerCallback_();
  }
}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolMallocInfoToMstx(void* addr, size_t size) {
  if (memoryMallocMstxCallback_) {
    memoryMallocMstxCallback_(addr, size);
  }
}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolFreeInfoToMstx(void* addr) {
  if (memoryFreeMstxCallback_) {
    memoryFreeMstxCallback_(addr);
  }
}

MemoryTimeEventPtr AbstractEnhancedDynamicMemPool::GenAllocateMemoryTimeEvent(
    const void* addr,
    size_t size,
    uint32_t streamId,
    bool fromPersistent,
    bool isPersistent) {
  auto timeEvent = std::make_shared<MemoryTimeEvent>();
  timeEvent->createdAt_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
          .count());
  timeEvent->addr_ = const_cast<void*>(addr);
  timeEvent->size_ = size;
  timeEvent->fromPersistent_ = static_cast<uint8_t>(fromPersistent);
  timeEvent->isPersistent_ = static_cast<uint8_t>(isPersistent);
  timeEvent->streamId_ = streamId;
  timeEvent->runMode_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().runMode_;
  timeEvent->usedSize_ = memStat_.usedSize_;
  timeEvent->peakSize_ = memStat_.peakSize_;
  timeEvent->allocSize_ = TotalMemStatistics();
  timeEvent->usedByEventSize_ = memStat_.usedByEventSize_;
  timeEvent->eagerFreeSize_ = memStat_.eagerFreeSize_;
  timeEvent->owner_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
  timeEvent->allocType_ = static_cast<uint8_t>(DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
  return timeEvent;
}

MemoryTimeEventPtr AbstractEnhancedDynamicMemPool::GenFreeMemoryTimeEvent(const void* addr) {
  auto timeEvent = std::make_shared<MemoryTimeEvent>();
  timeEvent->createdAt_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
          .count());
  timeEvent->addr_ = const_cast<void*>(addr);
  const size_t time_event_free_size = -1;
  timeEvent->size_ = time_event_free_size;
  timeEvent->usedSize_ = memStat_.usedSize_;
  timeEvent->peakSize_ = memStat_.peakSize_;
  timeEvent->allocSize_ = TotalMemStatistics();
  timeEvent->usedByEventSize_ = memStat_.usedByEventSize_;
  timeEvent->eagerFreeSize_ = memStat_.eagerFreeSize_;
  return timeEvent;
}
} // namespace device
} // namespace fxrt
