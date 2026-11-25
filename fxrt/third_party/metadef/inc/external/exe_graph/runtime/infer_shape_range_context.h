/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
#include <type_traits>
#include "range.h"
#include "runtime_tensor.h"
#include "runtime_attrs.h"
#include "extended_kernel_context.h"
namespace gert {
using TensorRange = Range<Tensor>;
/**
 * InferShapeRange kernel的context
 */
class InferShapeRangeContext : public ExtendedKernelContext {
 public:
  /**
   * 根据输入index，获取输入shape range指针
   * @param index 输入index
   * @return 输入shape range指针，index非法时，返回空指针
   */
  const Range<Shape>* GetInputShapeRange(const size_t index) const {
    return GetInputPointer<Range<Shape>>(index);
  }

  /**
   * 根据输入index，获取输出tensor指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param index 输入index
   * @return 输入tensor指针，index非法时，返回空指针
   */
  const TensorRange* GetInputTensorRange(const size_t index) const {
    return GetInputPointer<TensorRange>(index);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor range指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return tensor range指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const TensorRange* GetOptionalInputTensorRange(const size_t ir_index) const {
    return GetDynamicInputPointer<TensorRange>(ir_index, 0);
  }

  /**
   * 根据输入index，获取输入shape range指针
   * @param ir_index IR原型定义中的index
   * @return 输入shape range指针，ir_index非法时，返回空指针
   */
  const Range<Shape>* GetRequiredInputShapeRange(const size_t ir_index) const {
    return GetDynamicInputPointer<Range<Shape>>(ir_index, 0);
  }

  /**
   * 根据输入index，获取输出tensor指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return 输入tensor range指针，ir_index非法时，返回空指针
   */
  const TensorRange* GetRequiredInputTensorRange(const size_t ir_index) const {
    return GetDynamicInputPointer<TensorRange>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入tensor指针
   * 若算子被配置为'data'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return tensor range指针，index或relative_index非法时，返回空指针
   */
  const TensorRange* GetDynamicInputTensorRange(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<TensorRange>(ir_index, relative_index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape range指针
   * @param ir_index IR原型定义中的index
   * @return shape range指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Range<Shape>* GetOptionalInputShapeRange(const size_t ir_index) const {
    return GetDynamicInputPointer<Range<Shape>>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return shape range指针，index或relative_index非法时，返回空指针
   */
  const Range<Shape>* GetDynamicInputShapeRange(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Range<Shape>>(ir_index, relative_index);
  }

  /**
   * 根据输出index，获取输出shape range指针
   * @param index 输出index
   * @return 输出shape range指针，index非法时，返回空指针
   */
  Range<Shape>* GetOutputShapeRange(const size_t index) {
    return GetOutputPointer<Range<Shape>>(index);
  }
};
static_assert(std::is_standard_layout<InferShapeRangeContext>::value, "The class InferShapeContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_RANGE_CONTEXT_H_
