/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXTERNAL_EXE_GRAPH_RUNTIME_OP_EXECUTE_PREPARE_CONTEXT_H_
#define METADEF_CXX_INC_EXTERNAL_EXE_GRAPH_RUNTIME_OP_EXECUTE_PREPARE_CONTEXT_H_

#include "op_execute_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace fe {
class PlatFormInfos;
}

namespace gert {
enum class OpExecutePrepareInputExtendIndex {
  kExecuteOption,
  kFwkData, // 框架预留字段，算子开发者无需关注，新增字段向后追加
  kStream,
  // add new extend input here
  kNum
};

enum class OpExecutePrepareOutputIndex {
  kParams,
  kWorkspaceSize,
  // add new extend output here
  kNum
};

/**
 * Aclnn kernel的prepare context
 */
class OpExecutePrepareContext : public ExtendedKernelContext {
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
   * 获取确定性计算模式
   * @return bool，是否开启确定性计算, 异常情况默认返回false
   */
  bool GetDeterministic() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    const OpExecuteOptions* options = GetInputPointer<OpExecuteOptions>(
        input_num + output_num + static_cast<size_t>(OpExecutePrepareInputExtendIndex::kExecuteOption));
    if (options == nullptr) {
      return false;
    }
    return (options->deterministic != 0);
  }

  /**
   * 获取allow_hf32
   * @return string，是否开启hf32，正常情况返回 01，00，10，11四种字符串
   * 第一个字符表示Conv类算子是否支持hf32
   * 第二个字符表示MatMul类算子是否支持hf32，异常情况返回nullptr
   */
  const char* GetAllowHf32() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    const OpExecuteOptions* options = GetInputPointer<OpExecuteOptions>(
        input_num + output_num + static_cast<size_t>(OpExecutePrepareInputExtendIndex::kExecuteOption));
    if (options == nullptr) {
      return nullptr;
    }
    return options->allow_hf32;
  }

  /**
   * 获取精度模式
   * @return int32，精度模式，异常情况返回一个int32的极大值
   */
  int32_t GetPrecisionMode() const {
    const size_t input_num = GetComputeNodeInputNum();
    const size_t output_num = GetComputeNodeOutputNum();
    const OpExecuteOptions* options = GetInputPointer<OpExecuteOptions>(
        input_num + output_num + static_cast<size_t>(OpExecutePrepareInputExtendIndex::kExecuteOption));
    if (options == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    return options->precision_mode;
  }

  /**
   * 设置传递给Launch阶段的参数结构体
   * @param param 指向数据的指针
   * @return 设置结果
   */
  template <typename T>
  ge::graphStatus SetOpApiParamsWithDefaultDeleter(T* const params) {
    return SetOpApiParams(params, [](void* const data) { delete reinterpret_cast<T*>(data); });
  }

  /**
   * 设置传递给Launch阶段的参数结构体
   * @param param 指向数据的指针
   * @param deleter 释放数据的接口，必须非空；空指针或输出链不存在时返回ge::GRAPH_FAILED
   * @return 设置结果
   */
  ge::graphStatus SetOpApiParams(void* const params, const Chain::Deleter deleter) {
    auto params_chain = GetOutput(static_cast<size_t>(OpExecutePrepareOutputIndex::kParams));
    if (deleter == nullptr || params_chain == nullptr) {
      return ge::GRAPH_FAILED;
    }
    params_chain->Set(params, deleter);
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 设置workspace大小，会传递到Launch阶段
   * @param ws_sizes 设置的工作空间大小数组
   * @return 设置结果
   */
  ge::graphStatus SetWorkspaceSizes(const std::vector<uint64_t>& ws_sizes) {
    auto workspace_size_chain = GetOutput(static_cast<size_t>(OpExecutePrepareOutputIndex::kWorkspaceSize));
    if (workspace_size_chain == nullptr) {
      return ge::GRAPH_FAILED;
    }
    auto workspace_size = workspace_size_chain->GetPointer<TypedContinuousVector<size_t>>();
    if (workspace_size == nullptr) {
      auto size_vec = ContinuousVector::Create<size_t>(ws_sizes.size());
      workspace_size = reinterpret_cast<TypedContinuousVector<size_t>*>(size_vec.release());
      workspace_size_chain->SetWithDefaultDeleter<uint8_t[]>(workspace_size);
    }
    if (workspace_size->GetCapacity() < ws_sizes.size()) {
      return ge::GRAPH_FAILED;
    }
    workspace_size->SetSize(ws_sizes.size());
    auto* data = workspace_size->MutableData();
    for (size_t i = 0; i < ws_sizes.size(); ++i) {
      data[i] = static_cast<size_t>(ws_sizes[i]);
    }
    return ge::GRAPH_SUCCESS;
  }
};
static_assert(
    std::is_standard_layout<OpExecutePrepareContext>::value,
    "The class OpExecutePrepareContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXTERNAL_EXE_GRAPH_RUNTIME_OP_EXECUTE_PREPARE_CONTEXT_H_
