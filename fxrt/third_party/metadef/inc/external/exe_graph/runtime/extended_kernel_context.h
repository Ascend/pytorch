/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXTENDED_KERNEL_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXTENDED_KERNEL_CONTEXT_H_
#include <type_traits>
#include "kernel_context.h"
#include "context_extend.h"

namespace gert {
class ExtendedKernelContext : protected KernelContext {
 public:
  /**
   * 根据index获取节点输入的Tensor description
   * @param index 输入index
   * @return 输入TensorDesc的指针，index非法时，返回空指针
   */
  const CompileTimeTensorDesc* GetInputDesc(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetInputTdInfo(index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入CompileTimeTensorDesc指针
   * @param ir_index IR原型定义中的index
   * @return CompileTimeTensorDesc指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const CompileTimeTensorDesc* GetOptionalInputDesc(const size_t ir_index) const {
    return GetDynamicInputDesc(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入CompileTimeTensorDesc指针
   * @param ir_index IR原型定义中的index
   * @return CompileTimeTensorDesc指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const CompileTimeTensorDesc* GetRequiredInputDesc(const size_t ir_index) const {
    return GetDynamicInputDesc(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入CompileTimeTensorDesc指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return CompileTimeTensorDesc指针，index或relative_index非法时，返回空指针
   */
  const CompileTimeTensorDesc* GetDynamicInputDesc(const size_t ir_index, const size_t relative_index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    const auto ins_info = GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_index) {
      return nullptr;
    }
    return compute_node_info->GetInputTdInfo(ins_info->GetInstanceStart() + relative_index);
  }
  /**
   * 根据index获取节点输出的Tensor description
   * @param index 输出index
   * @return 输出TensorDesc的指针，index非法时，返回空指针
   */
  const CompileTimeTensorDesc* GetOutputDesc(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetOutputTdInfo(index);
  }
  /**
   * 根据IR原型中的index，获取在节点上的输入实例化信息
   * @param ir_index IR原型中定义的index
   * @return 实例化信息
   */
  const AnchorInstanceInfo* GetIrInputInstanceInfo(const size_t ir_index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetInputInstanceInfo(ir_index);
  }

  /**
   * 根据IR原型中的index，获取在节点上的输出实例化信息
   * @param ir_index IR原型中定义的index
   * @return 实例化信息
   */
  const AnchorInstanceInfo* GetIrOutputInstanceInfo(const size_t ir_index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetOutputInstanceInfo(ir_index);
  }

  /**
   * 获取计算节点的输入数量
   * @return 计算节点的输入数量
   */
  size_t GetComputeNodeInputNum() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return 0;
    }
    return compute_node_info->GetInputsNum();
  }
  /**
   * 获取计算节点的输出数量
   * @return 计算节点的输出数量
   */
  size_t GetComputeNodeOutputNum() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return 0;
    }
    return compute_node_info->GetOutputsNum();
  }
  /**
   * 获取计算节点的属性，仅IR原型中定义的属性可被获取到
   * @return 计算节点的属性
   */
  const RuntimeAttrs* GetAttrs() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetAttrs();
  }
  /**
   * 获取计算节点的类型
   * @return 计算节点的类型
   */
  const ge::char_t* GetNodeType() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetNodeType();
  }
  /**
   * 获取计算节点的name
   * @return 计算节点的name
   */
  const ge::char_t* GetNodeName() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    return compute_node_info->GetNodeName();
  }
  /**
   * 获取本kernel对应的计算节点的信息
   * @return 计算节点的信息
   */
  const ComputeNodeInfo* GetComputeNodeInfo() const {
    return static_cast<const ComputeNodeInfo*>(GetComputeNodeExtend());
  }
  /**
   * 获取本kernel的name
   * @return 本kernel的name
   */
  const ge::char_t* GetKernelName() const {
    const auto extend_info = GetExtendInfo();
    if (extend_info == nullptr) {
      return nullptr;
    }
    return extend_info->GetKernelName();
  }
  /**
   * 获取本kernel的type
   * @return 本kernel的type
   */
  const ge::char_t* GetKernelType() const {
    const auto extend_info = GetExtendInfo();
    if (extend_info == nullptr) {
      return nullptr;
    }
    return extend_info->GetKernelType();
  }
  /**
   * 获取本kernel的扩展信息
   * @return 本kernel的扩展信息
   */
  const KernelExtendInfo* GetExtendInfo() const {
    return static_cast<const KernelExtendInfo*>(GetKernelExtend());
  }

 protected:
  template <typename T, size_t Offset = 0>
  const T* GetDynamicInputPointer(size_t ir_index, size_t relative_ins_index) const {
    const auto ins_info = GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_ins_index) {
      return nullptr;
    }
    return GetInputPointer<T>(Offset + ins_info->GetInstanceStart() + relative_ins_index);
  }

  template <typename T, size_t Offset = 0>
  const T* GetDynamicOutputPointer(size_t ir_index, size_t relative_ins_index) const {
    const auto ins_info = GetIrOutputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_ins_index) {
      return nullptr;
    }
    const size_t input_num = GetComputeNodeInputNum();
    return GetInputPointer<T>(Offset + input_num + ins_info->GetInstanceStart() + relative_ins_index);
  }
};
static_assert(
    std::is_standard_layout<ExtendedKernelContext>::value,
    "The class ExtendedKernelRunContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXTENDED_KERNEL_CONTEXT_H_
