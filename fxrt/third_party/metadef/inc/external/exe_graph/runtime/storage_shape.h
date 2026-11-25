/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
#include <type_traits>
#include "shape.h"

namespace gert {
struct StorageShape {
 public:
  StorageShape() {
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra告警屏蔽
  };
  /**
   * 构造一个运行时shape实例
   * @param origin_shape 原始shape
   * @param storage_shape 运行时shape
   */
  StorageShape(const std::initializer_list<int64_t>& origin_shape, const std::initializer_list<int64_t>& storage_shape)
      : origin_shape_(origin_shape), storage_shape_(storage_shape) {
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra告警屏蔽
  }
  /**
   * 获取原始shape
   * @return 原始shape
   */
  const Shape& GetOriginShape() const {
    return origin_shape_;
  }
  /**
   * 获取shape
   * @return shape
   */
  const Shape& GetShape() const {
    return origin_shape_;
  }
  /**
   * 获取运行时shape
   * @return 运行时shape
   */
  const Shape& GetStorageShape() const {
    return storage_shape_;
  }
  /**
   * 获取可写的shape
   * @return 可写的shape
   */
  Shape& MutableShape() {
    return origin_shape_;
  }
  /**
   * 获取可写的原始shape
   * @return 可写的原始shape
   */
  Shape& MutableOriginShape() {
    return origin_shape_;
  }
  /**
   * 获取可写的运行时shape
   * @return 可写的运行时shape
   */
  Shape& MutableStorageShape() {
    return storage_shape_;
  }
  /**
   * 判断shape是否相等
   * @param other 另一个shape
   * @return true表示相等
   */
  bool operator==(const StorageShape& other) const {
    return origin_shape_ == other.origin_shape_ && storage_shape_ == other.storage_shape_;
  }
  /**
   * 判断shape是否不相等
   * @param other 另一个shape
   * @return true表示不相等
   */
  bool operator!=(const StorageShape& other) const {
    return !(*this == other);
  }

 private:
  Shape origin_shape_;
  Shape storage_shape_;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<StorageShape>::value, "The class must be a POD");
} // namespace gert

#endif // METADEF_CXX_INC_EXE_GRAPH_STORAGE_SHAPE_H_
