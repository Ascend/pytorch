/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXTERNAL_GE_ALLOCATOR_H_
#define METADEF_CXX_INC_EXTERNAL_GE_ALLOCATOR_H_
#include <cstdlib>
#include <memory>
namespace ge {
class MemBlock;
class Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() = default;
  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;

  virtual MemBlock* Malloc(size_t size) = 0;
  virtual void Free(MemBlock* block) = 0;

  // Apply for suggested address memory, default ignore suggested address
  virtual MemBlock* MallocAdvise(size_t size, void* addr) {
    (void)addr;
    return Malloc(size);
  }
};

class MemBlock {
 public:
  MemBlock(Allocator& allocator, void* addr, size_t block_size)
      : allocator_(allocator), addr_(addr), count_(1U), block_size_(block_size) {}
  virtual ~MemBlock() = default;
  const void* GetAddr() const {
    return addr_;
  }
  void* GetAddr() {
    return addr_;
  }
  size_t GetSize() const {
    return block_size_;
  }
  void SetSize(const size_t mem_size) {
    block_size_ = mem_size;
  }
  void Free() {
    if (GetCount() > 0U) {
      if (SubCount() == 0U) {
        return allocator_.Free(this);
      }
    }
  }

  size_t AddCount() {
    return ++count_;
  }
  size_t SubCount() {
    return --count_;
  }
  size_t GetCount() const {
    return count_;
  }

 private:
  Allocator& allocator_;
  void* addr_;
  size_t count_;
  size_t block_size_;
};

using AllocatorPtr = std::shared_ptr<Allocator>;
} // namespace ge
#endif // METADEF_CXX_INC_EXTERNAL_GE_ALLOCATOR_H_
