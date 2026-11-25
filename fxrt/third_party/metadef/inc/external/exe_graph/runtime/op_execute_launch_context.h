/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXTERNAL_EXE_GRAPH_RUNTIME_OP_EXECUTE_LAUNCH_CONTEXT_H_
#define METADEF_CXX_INC_EXTERNAL_EXE_GRAPH_RUNTIME_OP_EXECUTE_LAUNCH_CONTEXT_H_

#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
using rtStream = void*;
enum class OpExecuteLaunchInputIndex {
  kParams,
  kWorkspaceAddr,
  kWorkspaceSize,
  kStream,
  kFwkData, // 框架预留字段，算子开发者无需关注，新增字段向后追加
  // add new extend input here
  kNum
};

class OpExecuteLaunchContext : public ExtendedKernelContext {
 public:
  /**
   * 根据输入index，获取输出tensor指针
   * @param index 输入index
   * @return 输入tensor指针，index非法时，返回空指针
   */
  const Tensor* GetInputTensor(const size_t index) const {
    return GetInputPointer<Tensor>(index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor指针
   * @param ir_index IR原型定义中的index
   * @return tensor指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Tensor* GetOptionalInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入Tensor指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return tensor指针，index或relative_index非法时，返回空指针
   */
  const Tensor* GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }

  /**
   * 根据输出index，获取输出tensor指针
   * @param index 输出index
   * @return 输出tensor指针，index非法时，返回空指针
   */
  const Tensor* GetOutputTensor(const size_t index) const {
    if (index >= GetComputeNodeOutputNum()) {
      return nullptr;
    }
    const size_t input_num = GetComputeNodeInputNum();
    return GetInputPointer<Tensor>(input_num + index);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_OUTPUT`类型的输入Tensor指针
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_OUTPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return tensor指针，index或relative_index非法时，返回空指针
   */
  const Tensor* GetDynamicOutputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicOutputPointer<Tensor>(ir_index, relative_index);
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor指针
   * @param ir_index IR原型定义中的index
   * @return Tensor指针，index非法时，返回空指针
   */
  const Tensor* GetRequiredInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_OUTPUT`类型的输入Tensor指针
   * @param ir_index IR原型定义中的index
   * @return Tensor指针，index非法时，返回空指针
   */
  const Tensor* GetRequiredOutputTensor(const size_t ir_index) const {
    return GetDynamicOutputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 获取Prepare阶段传递给Launch阶段的参数结构体指针
   * @return void*, Param结构体指针
   */
  void* GetOpApiParams() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    return GetInputValue<void*>(input_num + output_num + static_cast<size_t>(OpExecuteLaunchInputIndex::kParams));
  }

  /**
   * 获取工作内存大小
   * @return TypedContinuousVector<size_t> *, 工作内存大小数组
   */
  const TypedContinuousVector<size_t>* GetWorkspaceSizes() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    return GetInputPointer<TypedContinuousVector<size_t>>(
        input_num + output_num + static_cast<size_t>(OpExecuteLaunchInputIndex::kWorkspaceSize));
  }

  /**
   * 获取工作内存地址
   * @return TypedContinuousVector<TensorData *>, 工作内存地址数组
   */
  const TypedContinuousVector<TensorData*>* GetWorkspaceAddrs() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    auto workspaces = GetInputPointer<TypedContinuousVector<TensorData*>>(
        input_num + output_num + static_cast<size_t>(OpExecuteLaunchInputIndex::kWorkspaceAddr));
    if (workspaces == nullptr || workspaces->GetSize() == 0) {
      return nullptr;
    }
    return workspaces;
  }

  /**
   * 获取stream
   * @return rtStream, aclnn算子下发的流, 异常情况返回nullptr
   */
  rtStream GetStream() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    return GetInputValue<rtStream>(input_num + output_num + static_cast<size_t>(OpExecuteLaunchInputIndex::kStream));
  }
};
static_assert(std::is_standard_layout<OpExecuteLaunchContext>::value, "The class OpExecuteLaunchContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXTERNAL_EXE_GRAPH_RUNTIME_OP_EXECUTE_LAUNCH_CONTEXT_H_
