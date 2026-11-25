/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXPAND_DIMS_TYPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXPAND_DIMS_TYPE_H_
#include <cstdint>
#include <cstddef>
#include "graph/error_codes.h"
#include "graph/types.h"
#include "shape.h"

namespace gert {
/**
 * 本类基于补维后的shape，描述了补维规则。本类设计为长度与一个uint64_t一致，因此可以以很小的代价做拷贝。
 *
 * 补维类似于ExpandDims算子，在原有shape的基础上，添加一到多个维度，例如原shape[2,2]有两根轴，那么在两根轴中间补两维后的shape为[2,1,1,2]。
 * 补维后shape的第0、3根轴被称为原始轴，第1、2根轴被称为补维轴。
 *
 * 本类通过1和0描述补维规则，1代表当前轴为补维轴，0代表当前轴为原始轴，从左到右依次代表当前shape每根轴的来源，例如：
 * | 补维规则 | 补维前shape | 补维后shape                                                  |
 * | -------- | ----------- | ------------------------------------------------------------ |
 * | 0110     | [2, 2]      | [2, 1, 1, 2]                                                 |
 * | 100      | [2, 3]      | [1, 2, 3]                                                    |
 * | 1000     | [2, 3]      | 补维规则与补维前shape不匹配，规则指定原始轴有3根，但原始shape只有2根轴，补维报错。 |
 *
 */
class ExpandDimsType {
 public:
  using AxisIndex = uint64_t;
  static constexpr size_t kMaxExpandSize = 56;

  ExpandDimsType() : size_(0U), mask_(0U) {
    (void)memset(reserved_, 0, sizeof(reserved_)); // memse函数misra告警屏蔽
  }

  /**
   * 通过字符串创建一个补维规则
   * @param expand_dims_type 字符串描述的补维规则
   */
  explicit ExpandDimsType(const ge::char_t* const expand_dims_type) : ExpandDimsType() {
    if (expand_dims_type == nullptr) {
      return;
    }
    const auto size = strlen(expand_dims_type);
    if (size > kMaxExpandSize) {
      // log error
      return;
    }

    size_ = size;
    for (AxisIndex i = 0; i < size; ++i) {
      if (expand_dims_type[i] == '1') {
        SetExpandIndex(i);
      }
    }
  }

  /**
   * 通过int创建一个补维规则，int中的位域定义为
   * | 字段     | 类型    | 含义                   |
   * | -------- | ------- | ---------------------- |
   * | 高8比特  | uint8_t | 补维规则长度           |
   * | 低56比特 | 位域    | 使用0、1描述的补维规则 |
   *
   * 为了实现简单，补维规则部分与字符串的顺序相反，例如字符串描述的补维规则为"1100"，那么对应的补维规则为"0011"转换为数字为3
   * @param expand_dims_type 补维规则
   */
  explicit ExpandDimsType(const int64_t reshape_type_mask) : ExpandDimsType() {
    if (reshape_type_mask == 0) {
      return;
    }
    size_ = static_cast<uint64_t>(static_cast<uint64_t>(reshape_type_mask) >> kMaxExpandSize);
    if (size_ > kMaxExpandSize) {
      return;
    }
    mask_ = static_cast<uint64_t>(static_cast<uint64_t>(reshape_type_mask) & 0x00FFFFFFFFFFFFFFULL);
  }
  /**
   * 判断补维规则是否一致
   * @param other 另一个实例
   * @return true/false
   */
  bool operator==(const ExpandDimsType& other) const {
    return (size_ == other.size_) && (mask_ == other.mask_);
  }
  /**
   * 获取补维后的dim数
   * @return
   */
  AxisIndex GetFullSize() const {
    return size_;
  }
  /**
   * 设置补维轴
   * @param index 第index根轴为补维轴
   */
  void SetExpandIndex(const AxisIndex index) {
    mask_ |= (1UL << index);
  }
  /**
   * 基于补维后的shape，判断index根轴是否为补维轴
   * @param index 第index根轴
   * @return true含义为补维轴，false含义为原shape的轴
   */
  bool IsExpandIndex(const AxisIndex index) const {
    return static_cast<bool>((1UL << index) & mask_);
  }
  /**
   * 对shape做补维，并将结果写入到out_shape
   * @param shape 输入shape，补维前shape
   * @param out_shape 输出shape，补维后shape
   * @return 补维成功返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus Expand(const Shape& shape, Shape& out_shape) const {
    if (shape.GetDimNum() == size_) {
      out_shape = shape;
      return ge::GRAPH_SUCCESS;
    }
    size_t shape_pos = 0;
    out_shape.SetDimNum(0);
    for (size_t out_shape_pos = 0; out_shape_pos < size_; ++out_shape_pos) {
      if (!IsExpandIndex(out_shape_pos)) {
        if (shape_pos >= shape.GetDimNum()) {
          return ge::GRAPH_FAILED;
        }
        (void)out_shape.AppendDim(shape.GetDim(shape_pos++));
      } else {
        (void)out_shape.AppendDim(1);
      }
    }

    for (; shape_pos < shape.GetDimNum(); ++shape_pos) {
      (void)out_shape.AppendDim(shape.GetDim(shape_pos));
    }
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 原地补维
   * @param shape 直接在本shape做补维
   * @return 补维成功返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus Expand(Shape& shape) const {
    // full_size:4, shape:[A,B], reshape_type:1010
    // shape:[A,B] + full_size:4 -> [A,B,1,1]
    if (shape.GetDimNum() == size_) {
      return ge::GRAPH_SUCCESS;
    }
    size_t dim_size = shape.GetDimNum();
    size_t cur_dim_size = dim_size;
    while (cur_dim_size++ < size_) {
      (void)shape.AppendDim(1);
    }

    // shape:[A,B,1,1] + 1010 -> [A,1,B,1]
    for (int32_t i = static_cast<int32_t>(size_ - 1UL); i >= 0; --i) {
      if (!IsExpandIndex(static_cast<AxisIndex>(i))) {
        if ((dim_size > 0) && (dim_size - 1UL < static_cast<size_t>(i))) {
          shape.SetDim(static_cast<size_t>(i), shape.GetDim(dim_size - 1));
          shape.SetDim(dim_size - 1UL, 1);
          dim_size--;
        }
      }
    }
    return ge::GRAPH_SUCCESS;
  }

 private:
  uint64_t size_ : 8;
  uint64_t mask_ : kMaxExpandSize;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};
} // namespace gert

#endif // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXPAND_DIMS_TYPE_H_
