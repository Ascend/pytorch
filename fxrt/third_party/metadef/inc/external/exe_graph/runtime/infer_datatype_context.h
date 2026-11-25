/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_DATATYPE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_DATATYPE_CONTEXT_H_
#include <type_traits>
#include "runtime_tensor.h"
#include "runtime_attrs.h"
#include "extended_kernel_context.h"
#include "graph/types.h"

namespace gert {
/**
 * InferDataType kernel的context
 */
class InferDataTypeContext : public ExtendedKernelContext {
 public:
  /**
   * 根据输入index，获取输入DataType
   * @param index 输入index
   * @return 输入datatype，index非法时，返回DT_UNDEFINED
   */
  ge::DataType GetInputDataType(const size_t index) const {
    const auto in_datatype = GetInputPointer<ge::DataType>(index);
    if (in_datatype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *in_datatype;
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入DataType
   * @param ir_index IR原型定义中的index
   * @return in_datatype，index非法，或该INPUT没有实例化时，返回DT_UNDEFINED
   */
  ge::DataType GetOptionalInputDataType(const size_t ir_index) const {
    const auto in_datatype = GetDynamicInputPointer<ge::DataType>(ir_index, 0);
    if (in_datatype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *in_datatype;
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入DataType
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return datatype，index或relative_index非法时，返回DT_UNDEFINED
   */
  ge::DataType GetDynamicInputDataType(const size_t ir_index, const size_t relative_index) const {
    const auto in_datatype = GetDynamicInputPointer<ge::DataType>(ir_index, relative_index);
    if (in_datatype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *in_datatype;
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入DataType
   * @param ir_index IR原型定义中的index
   * @return in_datatype，index非法，或该INPUT没有实例化时，返回DT_UNDEFINED
   */
  ge::DataType GetRequiredInputDataType(const size_t ir_index) const {
    const auto in_datatype = GetDynamicInputPointer<ge::DataType>(ir_index, 0);
    if (in_datatype == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *in_datatype;
  }

  /**
   * 根据输出index，获取输出DataType
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回DT_UNDEFINED
   */
  ge::DataType GetOutputDataType(const size_t index) const {
    const auto datatype_ptr = GetOutputPointer<ge::DataType>(index);
    if (datatype_ptr == nullptr) {
      return ge::DT_UNDEFINED;
    }
    return *datatype_ptr;
  }

  /**
   * 根据输出index，设置输出DataType
   * @param index 输出index
   * @param datatype 输出datatype
   * @return 设置结果，index非法时，返回失败
   */
  ge::graphStatus SetOutputDataType(const size_t index, const ge::DataType datatype) {
    const auto output_dtype = GetOutputPointer<ge::DataType>(index);
    if (output_dtype == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *output_dtype = datatype;
    return ge::GRAPH_SUCCESS;
  }
};
static_assert(std::is_standard_layout<InferDataTypeContext>::value, "The class InferDataTypeContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
