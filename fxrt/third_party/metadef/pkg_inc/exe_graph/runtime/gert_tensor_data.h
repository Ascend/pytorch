/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_GERT_TENSOR_DATA_H
#define METADEF_CXX_GERT_TENSOR_DATA_H
#include <utility>
#include "exe_graph/runtime/tensor_data.h"
#include "ge/ge_allocator.h"
#include "gert_mem_block.h"
namespace gert {
/**
 * 框架、引擎在执行时内部使用的TensorData，在TensorData的基础上，本类提供了引用计数和多流能力，但是这部分能力并不对外提供。
 * 因此在air仓外，仅基础的方法可使用，引用计数和多流能力(Share、Wander等方法)实现在air仓，仅链接air仓的引擎可调用
 */
class GertTensorData {
 public:
  GertTensorData()
      : tensor_data_{nullptr, nullptr, 0U, kTensorPlacementEnd}, gert_mem_block_{nullptr}, stream_id_{-1} {}

  /**
   * 有所有权构造函数
   * @param size
   * @param placement
   * @param stream_id
   * @param block
   */
  GertTensorData(size_t size, TensorPlacement placement, int64_t stream_id, GertMemBlock* block);

  /**
   * 无所有权构造函数
   * @param addr
   * @param size
   * @param placement
   * @param stream_id
   */
  GertTensorData(void* addr, size_t size, TensorPlacement placement, int64_t stream_id);

  GertTensorData(const GertTensorData&) = delete;
  GertTensorData& operator=(const GertTensorData& other) = delete;

  GertTensorData(GertTensorData&& other) noexcept
      : tensor_data_(std::move(other.tensor_data_)),
        gert_mem_block_(other.gert_mem_block_),
        stream_id_(other.stream_id_) {
    other.Clear();
  }

  GertTensorData& operator=(GertTensorData&& other) noexcept {
    if (this != &other) {
      Free();
      tensor_data_ = std::move(other.tensor_data_);
      gert_mem_block_ = other.gert_mem_block_;
      stream_id_ = other.stream_id_;
      other.Clear();
    }
    return *this;
  }
  ~GertTensorData() noexcept {
    Free();
  }

  size_t GetSize() const {
    return tensor_data_.GetSize();
  }
  void SetSize(const size_t size) {
    tensor_data_.SetSize(size);
  }

  TensorPlacement GetPlacement() const {
    return tensor_data_.GetPlacement();
  }
  void SetPlacement(const TensorPlacement placement) {
    tensor_data_.SetPlacement(placement);
  }

  TensorAddress GetAddr() const {
    return tensor_data_.GetAddr();
  }

  /*
   * 释放内存且清空地址
   */
  ge::graphStatus Free() {
    if (NeedFree()) {
      gert_mem_block_->Free(stream_id_);
      Clear();
    }
    return ge::GRAPH_SUCCESS;
  }

  /*
   * 释放内存但是依旧持有内存地址
   */
  ge::graphStatus FreeHoldAddr() {
    if (NeedFree()) {
      gert_mem_block_->Free(stream_id_);
      gert_mem_block_ = nullptr;
      stream_id_ = -1;
    }
    return ge::GRAPH_SUCCESS;
  }
  int64_t GetStreamId() const {
    return stream_id_;
  }

  const TensorData& GetTensorData() const {
    return tensor_data_;
  }
  TensorData& MutableTensorData() {
    return tensor_data_;
  }

  GertMemBlock* GetGertMemBlock() const {
    return gert_mem_block_;
  }

  GertMemBlock* Release() {
    auto block = gert_mem_block_;
    gert_mem_block_ = nullptr;
    return block;
  }

  bool IsSharedWith(const GertTensorData& other) const;
  ge::graphStatus ShareFrom(const GertTensorData& other);
  ge::graphStatus WanderFrom(const GertTensorData& other, int64_t dst_stream_id);

 private:
  void CopyFromWithoutStream(const GertTensorData& other);
  bool NeedFree() const {
    return gert_mem_block_ != nullptr;
  }
  void Clear() {
    tensor_data_.SetAddr(nullptr, nullptr);
    gert_mem_block_ = nullptr;
    stream_id_ = -1;
  }

 private:
  TensorData tensor_data_;
  GertMemBlock* gert_mem_block_{nullptr};
  int64_t stream_id_;
};
} // namespace gert
#endif // METADEF_CXX_GERT_TENSOR_DATA_H
