/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <securec.h>
#include "graph/error_codes.h"
#include "utils/extern_math_util.h"
#include "external/ge_common/api_error_codes.h"

namespace gert {
class ContinuousVector {
 public:
  /**
   * 创建一个ContinuousVector实例，ContinuousVector不支持动态扩容
   * @tparam T 实例中包含的元素类型
   * @param capacity 实例的最大容量
   * @param total_size 本实例的总长度
   * @return 指向本实例的指针
   */
  template <typename T>
  static std::unique_ptr<uint8_t[]> Create(size_t capacity, size_t& total_size) {
    if (ge::MulOverflow(capacity, sizeof(T), total_size)) {
      return nullptr;
    }
    if (ge::AddOverflow(total_size, sizeof(ContinuousVector), total_size)) {
      return nullptr;
    }
    auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
    if (holder == nullptr) {
      return nullptr;
    }
    reinterpret_cast<ContinuousVector*>(holder.get())->Init(capacity);
    return holder;
  }
  /**
   * 创建一个ContinuousVector实例，ContinuousVector不支持动态扩容
   * @tparam T 实例中包含的元素类型
   * @param capacity 实例的最大容量
   * @return 指向本实例的指针
   */
  template <typename T>
  static std::unique_ptr<uint8_t[]> Create(const size_t capacity) {
    size_t total_size;
    return Create<T>(capacity, total_size);
  }
  /**
   * 使用最大容量初始化本实例
   * @param capacity 最大容量
   */
  void Init(const size_t capacity) {
    capacity_ = capacity;
    size_ = 0U;
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }
  /**
   * 获取当前保存的元素个数
   * @return 当前保存的元素个数
   */
  size_t GetSize() const {
    return size_;
  }
  /**
   * 设置当前保存的元素个数
   * @param size 当前保存的元素个数
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetSize(const size_t size) {
    if (size > capacity_) {
      return ge::GRAPH_PARAM_OUT_OF_RANGE;
    }
    size_ = size;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取最大可保存的元素个数
   * @return 最大可保存的元素个数
   */
  size_t GetCapacity() const {
    return capacity_;
  }
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  const void* GetData() const {
    return elements;
  }
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  void* MutableData() {
    return elements;
  }

 private:
  size_t capacity_;
  size_t size_;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
  uint8_t elements[8];
};
static_assert(std::is_standard_layout<ContinuousVector>::value, "The ContinuousVector must be a POD");

template <typename T>
class TypedContinuousVector : private ContinuousVector {
 public:
  using ContinuousVector::GetCapacity;
  using ContinuousVector::GetSize;
  using ContinuousVector::SetSize;
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  T* MutableData() {
    return static_cast<T*>(ContinuousVector::MutableData());
  }
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  const T* GetData() const {
    return static_cast<const T*>(ContinuousVector::GetData());
  }
};

/*
 * memory layout: |size_|offset1|offset2|...|ContinuousVector1|ContinuousVector1|...|
 * size_ : number of ContinuousVector
 * offset1 : offset of ContinuousVector1u
 */
class ContinuousVectorVector {
 public:
  void Init(const size_t capacity) {
    capacity_ = capacity;
    size_ = 0U;
    if (capacity_ == 0U) {
      return;
    }
    SetOffset(0U, GetOverHeadLength(capacity_));
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  template <typename T>
  ContinuousVector* Add(size_t inner_vector_capacity) {
    if (size_ >= capacity_) {
      return nullptr;
    }
    const auto inner_vector = reinterpret_cast<ContinuousVector*>(reinterpret_cast<uint8_t*>(this) + GetOffset(size_));
    inner_vector->Init(inner_vector_capacity);
    (void)inner_vector->SetSize(inner_vector_capacity);
    size_t inner_vector_length = 0U;
    if (ge::MulOverflow(inner_vector_capacity, sizeof(T), inner_vector_length)) {
      return nullptr;
    }
    if (ge::AddOverflow(inner_vector_length, sizeof(ContinuousVector), inner_vector_length)) {
      return nullptr;
    }
    ++size_;
    if (size_ < capacity_) {
      SetOffset(size_, GetOffset(size_ - 1U) + inner_vector_length);
    }
    return inner_vector;
  }

  const ContinuousVector* Get(const size_t index) const {
    return reinterpret_cast<const ContinuousVector*>(reinterpret_cast<const uint8_t*>(this) + GetOffset(index));
  }

  size_t GetSize() const {
    return size_;
  }

  static size_t GetOverHeadLength(const size_t capacity) {
    if (capacity == 0U) {
      return sizeof(ContinuousVectorVector);
    }
    return sizeof(capacity_) + sizeof(size_) + sizeof(reserved_) + sizeof(size_t) * capacity;
  }

 private:
  void SetOffset(const size_t index, const size_t offset) {
    size_t* const offset_ptr = &offset_[0U];
    offset_ptr[index] = offset;
  }

  size_t GetOffset(const size_t index) const {
    const size_t* const offset_ptr = &offset_[0U];
    return offset_ptr[index];
  }

 private:
  size_t capacity_ = 0U;
  size_t size_ = 0U;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
  size_t offset_[1U];
};
static_assert(std::is_standard_layout<ContinuousVectorVector>::value, "The ContinuousVectorVector must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
