/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_INFER_FORMAT_CONTEXT_H_
#define METADEF_CXX_INC_GRAPH_INFER_FORMAT_CONTEXT_H_
#include <type_traits>
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/runtime_tensor.h"

namespace gert {
class InferFormatContext : public ExtendedKernelContext {
 public:
  /**
   * 根据输入index，获取输入format指针
   * @param index 输入index
   * @return 输入format指针，index非法时，返回空指针
   */
  StorageFormat* GetInputFormat(const size_t index) {
    auto tensor = MutableInputPointer<Tensor>(index);
    if (tensor == nullptr) {
      return nullptr;
    }
    return &(tensor->MutableFormat());
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入format指针
   * @param ir_index IR原型定义中的index
   * @return 输入format指针，ir_index非法时，返回空指针
   */
  StorageFormat* GetRequiredInputFormat(const size_t ir_index) {
    auto tensor = GetDynamicInputTensorImpl(ir_index, 0);
    if (tensor == nullptr) {
      return nullptr;
    }
    return &(tensor->MutableFormat());
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入format指针
   * @param ir_index IR原型定义中的index
   * @return 输入format指针，ir_index非法，或该INPUT没有实例化时，返回空指针
   */
  StorageFormat* GetOptionalInputFormat(const size_t ir_index) {
    auto tensor = GetDynamicInputTensorImpl(ir_index, 0);
    if (tensor == nullptr) {
      return nullptr;
    }
    return &(tensor->MutableFormat());
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入format指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return 输入format指针，ir_index或relative_index非法时，返回空指针
   */
  StorageFormat* GetDynamicInputFormat(const size_t ir_index, const size_t relative_index) {
    const auto tensor = GetDynamicInputTensorImpl(ir_index, relative_index);
    if (tensor == nullptr) {
      return nullptr;
    }
    return &(tensor->MutableFormat());
  }

  /**
   * 根据输入index，获取输入shape指针
   * @param index 输入index
   * @return 输入shape指针，index非法时，返回空指针
   */
  const Shape* GetInputShape(const size_t index) const {
    return GetInputPointer<Shape>(index);
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @return 输入shape指针，ir_index非法时，返回空指针
   */
  const Shape* GetRequiredInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @return 输入shape指针，ir_index非法，或该INPUT没有实例化时，返回空指针
   */
  const Shape* GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return 输入shape指针，ir_index或relative_index非法时，返回空指针
   */
  const Shape* GetDynamicInputShape(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, relative_index);
  }

  /**
   * 根据输入index，获取输入tensor指针
   * @param index 输入index
   * @return 输入tensor指针，index非法时，返回空指针
   */
  const Tensor* GetInputTensor(const size_t index) const {
    return GetInputPointer<Tensor>(index);
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor指针
   * @param ir_index IR原型定义中的index
   * @return 输入tensor指针，ir_index非法时，返回空指针
   */
  const Tensor* GetRequiredInputTensor(const size_t ir_index) const {
    return GetDynamicInputTensorImpl(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor指针
   * @param ir_index IR原型定义中的index
   * @return 输入tensor指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Tensor* GetOptionalInputTensor(const size_t ir_index) const {
    return GetDynamicInputTensorImpl(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入tensor指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return 输入tensor指针，ir_index或relative_index非法时，返回空指针
   */
  const Tensor* GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputTensorImpl(ir_index, relative_index);
  }

  /**
   * 根据输出index，获取输出format指针
   * @param index 输出index
   * @return 输出format指针，index非法时，返回空指针
   */
  StorageFormat* GetOutputFormat(const size_t index) {
    const auto tensor = GetOutputPointer<Tensor>(index);
    if (tensor == nullptr) {
      return nullptr;
    }
    return &(tensor->MutableFormat());
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_OUTPUT`类型的输出format指针
   * @param ir_index IR原型定义中的index
   * @return 输出format指针，ir_index非法时，返回空指针
   */
  StorageFormat* GetRequiredOutputFormat(const size_t ir_index) {
    const auto tensor = GetDynamicOutputTensorImpl(ir_index, 0);
    if (tensor == nullptr) {
      return nullptr;
    }
    return &(tensor->MutableFormat());
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_OUTPUT`类型的输出format指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_OUTPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return 输出format指针，ir_index或relative_index非法时，返回空指针
   */
  StorageFormat* GetDynamicOutputFormat(const size_t ir_index, const size_t relative_index) {
    const auto tensor = GetDynamicOutputTensorImpl(ir_index, relative_index);
    if (tensor == nullptr) {
      return nullptr;
    }
    return &(tensor->MutableFormat());
  }

 private:
  Tensor* GetDynamicInputTensorImpl(size_t ir_index, size_t relative_ins_index) const {
    const auto ins_info = GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_ins_index) {
      return nullptr;
    }
    return MutableInputPointer<Tensor>(ins_info->GetInstanceStart() + relative_ins_index);
  }

  Tensor* GetDynamicOutputTensorImpl(size_t ir_index, size_t relative_ins_index) {
    const auto ins_info = GetIrOutputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_ins_index) {
      return nullptr;
    }
    return GetOutputPointer<Tensor>(ins_info->GetInstanceStart() + relative_ins_index);
  }
};
static_assert(std::is_standard_layout<InferFormatContext>::value, "The class InferFormatContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_GRAPH_INFER_FORMAT_CONTEXT_H_
