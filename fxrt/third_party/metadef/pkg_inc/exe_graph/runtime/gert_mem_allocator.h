/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_GERT_MEM_ALLOCATOR_H
#define METADEF_CXX_GERT_MEM_ALLOCATOR_H
#include "gert_mem_block.h"
#include "gert_tensor_data.h"
#include "ge/ge_allocator.h"
namespace gert {
class GertAllocator {
 public:
  GertAllocator() : GertAllocator(-1, kTensorPlacementEnd) {}
  GertAllocator(int64_t stream_id, TensorPlacement placement) : stream_id_(stream_id), placement_(placement) {}
  GertAllocator(const GertAllocator&) = delete;
  GertAllocator(GertAllocator&&) = delete;
  GertAllocator& operator=(const GertAllocator&) = delete;
  GertAllocator& operator=(GertAllocator&&) = delete;

  virtual ~GertAllocator() = default;
  virtual GertMemBlock* Malloc(size_t size) = 0;
  virtual GertTensorData MallocTensorData(size_t size) = 0;
  virtual TensorData MallocTensorDataFromL1(size_t size) = 0;

  virtual void Free(GertMemBlock* block) = 0;
  virtual ge::graphStatus FreeAt(int64_t stream_id, GertMemBlock* block) = 0;

  virtual ge::graphStatus ShareFromTensorData(const TensorData& td, GertTensorData& gtd) = 0;

  virtual int64_t GetStreamNum() = 0;

  virtual ge::graphStatus SetL1Allocator(ge::Allocator* allocator) = 0;

  virtual ge::graphStatus MoveL2ToL1(GertMemBlock* block) {
    (void)block;
    return ge::GRAPH_SUCCESS;
  }

  TensorPlacement GetPlacement() const {
    return placement_;
  }
  void SetPlacement(TensorPlacement placement) {
    placement_ = placement;
  }
  void SetStreamId(int64_t stream_id) {
    stream_id_ = stream_id;
  }
  int64_t GetStreamId() const {
    return stream_id_;
  }

 private:
  int64_t stream_id_;
  TensorPlacement placement_;
};
} // namespace gert
#endif // METADEF_CXX_GERT_MEM_ALLOCATOR_H
