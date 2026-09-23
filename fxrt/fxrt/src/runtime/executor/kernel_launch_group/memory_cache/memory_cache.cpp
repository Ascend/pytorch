#include "runtime/executor/kernel_launch_group/memory_cache/memory_cache.h"
#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <unordered_map>

namespace fxrt {
namespace runtime {
MemoryCache::MemoryCache() {
  kernelMemoryTraceBlocks_ = std::make_shared<std::map<const DeviceContext*, std::vector<KernelMemoryTraceBlockPtr>>>();
  mergedMemoryTraceBlocks_ = std::make_shared<std::map<const DeviceContext*, std::vector<MemoryTraceBlockPtr>>>();
  kernelToBlock_ = std::make_shared<std::unordered_map<OpRunner*, std::vector<KernelMemoryTraceBlockPtr>>>();
  tensorToKernelMemBlocks_ =
      std::make_shared<std::unordered_map<const Tensor*, KernelMemoryTraceBlockPtr>>(); // NOLINT(whitespace/indent)
}

void MemoryCache::ReserveKernelMemoryBlocks(size_t size, const DeviceContext* deviceContext) {
  CHECK_IF_NULL(deviceContext);
  (*kernelMemoryTraceBlocks_)[deviceContext].reserve(size);
}

void MemoryCache::AddKernelMemoryTraceBlock(
    const KernelMemoryTraceBlockPtr& block,
    const DeviceContext* deviceContext) {
  CHECK_IF_NULL(block);
  CHECK_IF_NULL(block->start_);
  CHECK_IF_NULL(block->end_);
  (*kernelMemoryTraceBlocks_)[deviceContext].emplace_back(block);
}

const std::shared_ptr<std::map<const DeviceContext*, std::vector<MemoryTraceBlockPtr>>>& MemoryCache::GetMergeBlocks()
    const {
  return mergedMemoryTraceBlocks_;
}

const std::shared_ptr<std::unordered_map<OpRunner*, std::vector<KernelMemoryTraceBlockPtr>>>& MemoryCache::
    GetAllKernelBlocksInfo() {
  return kernelToBlock_;
}

const std::shared_ptr<std::unordered_map<const Tensor*, KernelMemoryTraceBlockPtr>>& MemoryCache::
    GetTensorToMemBlocksInfo() const {
  return tensorToKernelMemBlocks_;
}

void MemoryCache::MergeBlocks() {
  mergedMemoryTraceBlocks_->clear();
  for (auto& item : *kernelMemoryTraceBlocks_) {
    auto& deviceContext = item.first;
    auto& kernelMemoryTraceBlocks = item.second;
    MergeBlocksForSameDeviceContext(&kernelMemoryTraceBlocks, &((*mergedMemoryTraceBlocks_)[deviceContext]));
    RT_VLOG(VL_RUNTIME) << "The number of merged blocks is " << (*mergedMemoryTraceBlocks_)[deviceContext].size()
                        << ", device: " << deviceContext->GetDeviceContextKey().deviceName_;
  }
}

void MemoryCache::MergeBlocksForSameDeviceContext(
    std::vector<KernelMemoryTraceBlockPtr>* kernelMemoryTraceBlocks,
    std::vector<MemoryTraceBlockPtr>* mergedMemoryTraceBlocks) {
  CHECK_IF_NULL(kernelMemoryTraceBlocks);
  CHECK_IF_NULL(mergedMemoryTraceBlocks);
  mergedMemoryTraceBlocks->clear();

  if (kernelMemoryTraceBlocks->empty()) {
    RT_VLOG(VL_RUNTIME) << "No block to merge.";
    return;
  }

  std::sort(
      kernelMemoryTraceBlocks->begin(),
      kernelMemoryTraceBlocks->end(),
      [](const KernelMemoryTraceBlockPtr& block1, const KernelMemoryTraceBlockPtr& block2) {
        return (block1->start_ < block2->start_) ||
            ((block1->start_ == block2->start_) && (block1->end_ < block2->end_));
      });
  mergedMemoryTraceBlocks->emplace_back(
      std::make_shared<MemoryTraceBlock>((*kernelMemoryTraceBlocks)[0]->start_, (*kernelMemoryTraceBlocks)[0]->size_));
  (*kernelMemoryTraceBlocks)[0]->memoryTraceBlockIndex_ = 0;
  for (size_t i = 1; i < kernelMemoryTraceBlocks->size(); i++) {
    auto& back = mergedMemoryTraceBlocks->back();
    auto& block = (*kernelMemoryTraceBlocks)[i];
    if (block->start_ >= back->end_) {
      mergedMemoryTraceBlocks->emplace_back(std::make_shared<MemoryTraceBlock>(block->start_, block->size_));
    } else if (block->end_ > back->end_) {
      back->end_ = block->end_;
      back->size_ = back->end_ - back->start_;
    }
    block->memoryTraceBlockIndex_ = mergedMemoryTraceBlocks->size() - 1;
  }

  // Reset offset
  for (size_t i = 0; i < kernelMemoryTraceBlocks->size(); i++) {
    auto& kernelMemBlock = (*kernelMemoryTraceBlocks)[i];
    CHECK_IF_NULL(kernelMemBlock);
    const auto& memBlock = (*mergedMemoryTraceBlocks)[kernelMemBlock->memoryTraceBlockIndex_];
    CHECK_IF_NULL(memBlock);
    if (kernelMemBlock->start_ < memBlock->start_) {
      RT_GLOG(EXCEPTION) << "Invalid memory block, block start: " << kernelMemBlock->start_
                         << ", block end: " << kernelMemBlock->end_ << ", mem block start: " << memBlock->start_
                         << ", mem block end: " << memBlock->end_;
    }

    kernelMemBlock->offsetInMemoryTraceBlock_ = kernelMemBlock->start_ - memBlock->start_;
    (*kernelToBlock_)[kernelMemBlock->op_].emplace_back(kernelMemBlock);
    if (kernelMemBlock->memType_ == kOutputMem) {
      tensorToKernelMemBlocks_->emplace(kernelMemBlock->tensor_, kernelMemBlock);
    }
  }
}

void MemoryCache::ClearExpiredCache() {
  if (kernelMemoryTraceBlocks_) {
    kernelMemoryTraceBlocks_->clear();
  }

  if (mergedMemoryTraceBlocks_) {
    mergedMemoryTraceBlocks_->clear();
  }

  if (kernelToBlock_) {
    kernelToBlock_->clear();
  }

  if (tensorToKernelMemBlocks_) {
    tensorToKernelMemBlocks_->clear();
  }
}

void MemoryCache::ClearAllCache() {
  if (kernelMemoryTraceBlocks_) {
    kernelMemoryTraceBlocks_->clear();
    kernelMemoryTraceBlocks_ = nullptr;
  }

  if (mergedMemoryTraceBlocks_) {
    mergedMemoryTraceBlocks_->clear();
    mergedMemoryTraceBlocks_ = nullptr;
  }

  if (kernelToBlock_) {
    kernelToBlock_->clear();
    kernelToBlock_ = nullptr;
  }

  if (tensorToKernelMemBlocks_) {
    tensorToKernelMemBlocks_->clear();
    tensorToKernelMemBlocks_ = nullptr;
  }
}
} // namespace runtime
} // namespace fxrt
