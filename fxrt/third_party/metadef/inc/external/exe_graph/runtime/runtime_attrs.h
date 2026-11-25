/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATTRS_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATTRS_H_
#include <cstdint>
#include <type_traits>
#include <cstddef>
#include "continuous_vector.h"
#include "runtime_tensor.h"

namespace gert {
class RuntimeAttrs {
 public:
  /**
   * 获取属性
   * @tparam T 属性类型
   * @param index 属性index
   * @return 指向属性的指针
   */
  template <typename T>
  const T* GetAttrPointer(size_t index) const {
    return reinterpret_cast<const T*>(GetPointerByIndex(index));
  }
  /**
   * 获取int类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const int64_t* GetInt(const size_t index) const {
    return GetAttrPointer<int64_t>(index);
  }
  /**
   * 获取list int类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const TypedContinuousVector<int64_t>* GetListInt(const size_t index) const {
    return GetAttrPointer<TypedContinuousVector<int64_t>>(index);
  }
  /**
   * 获取list int类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const ContinuousVectorVector* GetListListInt(const size_t index) const {
    return GetAttrPointer<ContinuousVectorVector>(index);
  }
  /**
   * 获取string类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const char* GetStr(const size_t index) const {
    return GetAttrPointer<char>(index);
  }
  /**
   * 获取tensor类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const Tensor* GetTensor(const size_t index) const {
    return GetAttrPointer<Tensor>(index);
  }
  /**
   * 获取float类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const float* GetFloat(const size_t index) const {
    return GetAttrPointer<float>(index);
  }
  /**
   * 获取bool类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const bool* GetBool(const size_t index) const {
    return GetAttrPointer<bool>(index);
  }

  /**
   * 获取list_float32类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const TypedContinuousVector<float>* GetListFloat(const size_t index) const {
    return GetAttrPointer<TypedContinuousVector<float>>(index);
  }

  /**
   * 获取list_float32类型属性
   * @param index 属性index
   * @return 指向属性值的指针
   */
  const ContinuousVectorVector* GetListListFloat(const size_t index) const {
    return GetAttrPointer<ContinuousVectorVector>(index);
  }

  /**
   * 获取属性数量
   * @return 属性数量
   */
  size_t GetAttrNum() const;

  RuntimeAttrs() = delete;
  RuntimeAttrs(const RuntimeAttrs&) = delete;
  RuntimeAttrs(RuntimeAttrs&&) = delete;
  RuntimeAttrs& operator=(const RuntimeAttrs&) = delete;
  RuntimeAttrs& operator=(RuntimeAttrs&&) = delete;

 private:
  const void* GetPointerByIndex(size_t index) const;

  uint64_t placeholder_;
};
static_assert(std::is_standard_layout<RuntimeAttrs>::value, "This class must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATTRS_H_
