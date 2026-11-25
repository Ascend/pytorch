/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_

#include <array>
#include <vector>
#include <iostream>
#include <cstring>
#include <type_traits>
#include <limits>
#include "utils/extern_math_util.h"

namespace gert {
struct Shape {
 public:
  static constexpr size_t kMaxDimNum = 25;
  static constexpr int64_t kInvalidDimValue = std::numeric_limits<int64_t>::min();

 public:
  /**
   * 默认构造一个shape，默认构造的shape实例中，dim_num长度为0
   */
  Shape() : dim_num_(0), dims_{0} {
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra告警屏蔽
  }

  /**
   * 通过dims值构造shape，例如：Shape({8,3,224,224})创建一个Shape实例，有4个维度，每个维度的值分别是8,3,224,224
   * @param dims shape的所有dim值
   */
  Shape(const std::initializer_list<int64_t>& args) : Shape() {
    if (args.size() > kMaxDimNum) {
      return;
    }
    dim_num_ = args.size();
    size_t i = 0;
    for (const int64_t arg : args) {
      dims_[i++] = arg;
    }
  }

  /**
   * 拷贝构造
   * @param other 源对象
   * 为了提升性能，dims_超过dim_num_的空间没有拷贝，可能有脏数据
   */
  Shape(const Shape& other) {
    dim_num_ = other.dim_num_;
    for (size_t i = 0U; i < dim_num_; ++i) {
      dims_[i] = other.dims_[i];
    }
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra告警屏蔽
  }

  /**
   * 拷贝赋值
   * @param other
   * @return
   * 为了提升性能，dims_超过dim_num_的空间没有拷贝，可能有脏数据
   */
  Shape& operator=(const Shape& other) {
    if (&other != this) {
      dim_num_ = other.dim_num_;
      for (size_t i = 0U; i < dim_num_; ++i) {
        dims_[i] = other.dims_[i];
      }
    }
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra告警屏蔽
    return *this;
  }

  /**
   * 判断与另外一个shape对象是否相等，如果两个shape的dim num并且dim num内每个dim的值都相等，那么认为两个shape相等
   * @param rht 另一个Shape对象
   * @return true/false
   */
  bool operator==(const Shape& rht) const {
    if (this->dim_num_ != rht.dim_num_) {
      return false;
    }
    for (size_t i = 0; i < this->dim_num_; i++) {
      if (this->dims_[i] != rht.dims_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * 判断与另一个Shape对象是否不等
   * @param rht 另一个Shape对象
   * @return true/false
   */
  bool operator!=(const Shape& rht) const {
    return !(*this == rht);
  }

  /**
   * 获取shape size，所谓shape size，是指shape中有多少个元素
   * @return shape-size，在发生溢出时，返回`kInvalidDimValue`
   */
  int64_t GetShapeSize() const {
    for (size_t i = 0; i < dim_num_; ++i) {
      if (dims_[i] == 0) {
        return 0;
      }
    }

    int64_t shape_size = 1;
    for (size_t i = 0; i < dim_num_; ++i) {
      if (ge::MulOverflow(shape_size, dims_[i], shape_size)) {
        return kInvalidDimValue;
      }
    }
    return shape_size;
  }

  /**
   * 判断本shape是否为标量，所谓标量，是指GetDimNum()为0
   * @return true/false
   */
  bool IsScalar() const {
    return dim_num_ == 0L;
  }

  /**
   * 设置shape为标量
   * @param none
   */
  void SetScalar() {
    dim_num_ = 0;
  }

  /**
   * 获取dim num
   * @return
   */
  size_t GetDimNum() const {
    return dim_num_;
  }

  /**
   * 设置dim num
   * @param dim_num
   * 超出最大维度数时保持当前dim num不变
   */
  void SetDimNum(const size_t dim_num) {
    if (dim_num > kMaxDimNum) {
      return;
    }
    this->dim_num_ = dim_num;
  }

  /**
   * 获取dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return dim值，在idx超出MaxDimNum时，返回`kInvalidDimValue`
   */
  int64_t GetDim(const size_t idx) const {
    if (idx >= kMaxDimNum) {
      return kInvalidDimValue;
    }
    return dims_[idx];
  }

  /**
   * 获取dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return dim值，行为未定义
   */
  const int64_t& operator[](const size_t idx) const {
    return dims_[idx];
  }

  /**
   * 获取dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return dim值，在idx超出MaxDimNum时，行为未定义
   */
  int64_t& operator[](const size_t idx) {
    return dims_[idx];
  }

  /**
   * 设置dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return
   */
  void SetDim(size_t idx, const int64_t dim_value) {
    if (idx >= kMaxDimNum) {
      return;
    }
    dims_[idx] = dim_value;
  }

  /**
   * 向后扩展一个dim值，如果扩展的dim数量超出Shape的最大限制，那么本函数不做任何事情
   * @param 扩展的dim值
   * @return this引用
   */
  Shape& AppendDim(const int64_t value) {
    if (this->dim_num_ >= kMaxDimNum) {
      return *this;
    }
    dims_[this->dim_num_++] = value;
    return *this;
  }

 private:
  size_t dim_num_;
  int64_t dims_[kMaxDimNum];
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<Shape>::value, "The class Shape must be a POD");
} // namespace gert

#endif // METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_
