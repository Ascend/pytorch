/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_STRIDE_H_
#define METADEF_CXX_INC_EXE_GRAPH_STRIDE_H_

#include <array>
#include <vector>
#include <iostream>
#include <cstring>
#include <type_traits>
#include <limits>
#include "utils/extern_math_util.h"

namespace gert {
struct Stride {
 public:
  static constexpr size_t kMaxDimNum = 25;
  static constexpr int64_t kInvalidStrideValue = std::numeric_limits<int64_t>::min();

 public:
  /**
   * 默认构造一个Stride，默认构造的Stride实例中，dim_num长度为0
   */
  Stride() : dim_num_(0), strides_{0} {
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra屏蔽
  }

  /**
   * 通过stride值构造Stride，例如：Stride({2,4,8,16})创建一个Stride实例，有4个维度，每个维度的步长分别是2,4,8,16
   * @param strides Stride的所有维度的步长
   */
  Stride(const std::initializer_list<int64_t>& args) : Stride() {
    if (args.size() > kMaxDimNum) {
      return;
    }
    dim_num_ = args.size();
    size_t i = 0;
    for (const int64_t arg : args) {
      strides_[i++] = arg;
    }
  }

  /**
   * 拷贝构造
   * @param other 源对象
   * 为了提升性能，strides_超过dim_num_的空间没有拷贝，可能有脏数据
   */
  Stride(const Stride& other) {
    dim_num_ = other.dim_num_;
    for (size_t i = 0U; i < dim_num_; ++i) {
      strides_[i] = other.strides_[i];
    }
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra屏蔽
  }

  /**
   * 拷贝赋值
   * @param other
   * @return
   * 为了提升性能，strides_超过dim_num_的空间没有拷贝，可能有脏数据
   */
  Stride& operator=(const Stride& other) {
    if (&other != this) {
      dim_num_ = other.dim_num_;
      for (size_t i = 0U; i < dim_num_; ++i) {
        strides_[i] = other.strides_[i];
      }
    }
    (void)memset(reserved_, 0, sizeof(reserved_)); // memset函数misra屏蔽
    return *this;
  }

  /**
   * 判断与另外一个Stride对象是否相等，如果两个Stride的dim num相等并且dim
   * num内每个stride的值都相等，那么认为两个Stride相等
   * @param rht 另一个Stride对象
   * @return true/false
   */
  bool operator==(const Stride& rht) const {
    if (this->dim_num_ != rht.dim_num_) {
      return false;
    }
    for (size_t i = 0; i < this->dim_num_; i++) {
      if (this->strides_[i] != rht.strides_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * 判断与另一个Stride对象是否不等
   * @param rht 另一个Stride对象
   * @return true/false
   */
  bool operator!=(const Stride& rht) const {
    return !(*this == rht);
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
   * 获取stride值
   * @param idx stride的index，调用者需要保证index合法
   * @return stride值，在idx超出MaxDimNum时，返回`kInvalidStrideValue`
   */
  int64_t GetStride(const size_t idx) const {
    if (idx >= kMaxDimNum) {
      return kInvalidStrideValue;
    }
    return strides_[idx];
  }

  /**
   * 获取stride值
   * @param idx stride的index，调用者需要保证index合法
   * @return stride值，行为未定义
   */
  const int64_t& operator[](const size_t idx) const {
    return strides_[idx];
  }

  /**
   * 获取stride值
   * @param idx stride的index，调用者需要保证index合法
   * @return stride值，在idx超出MaxDimNum时，行为未定义
   */
  int64_t& operator[](const size_t idx) {
    return strides_[idx];
  }

  /**
   * 设置stride值
   * @param idx stride的index，调用者需要保证index合法
   * @return
   */
  void SetStride(size_t idx, const int64_t stride) {
    if (idx >= kMaxDimNum) {
      return;
    }
    strides_[idx] = stride;
  }

  /**
   * 向后扩展一个stride值，如果扩展的dim数量超出Stride的最大限制，那么本函数不做任何事情
   * @param 扩展的stride值
   * @return this引用
   */
  Stride& AppendStride(const int64_t value) {
    if (this->dim_num_ >= kMaxDimNum) {
      return *this;
    }
    strides_[this->dim_num_++] = value;
    return *this;
  }

 private:
  size_t dim_num_;
  int64_t strides_[kMaxDimNum];
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};
static_assert(std::is_standard_layout<Stride>::value, "The class Stride must be a POD");
} // namespace gert

#endif // METADEF_CXX_INC_EXE_GRAPH_STRIDE_H_
