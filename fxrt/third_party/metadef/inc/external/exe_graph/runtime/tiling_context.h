/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#include "storage_shape.h"
#include "runtime_tensor.h"
#include "continuous_vector.h"
#include "extended_kernel_context.h"
#include "tiling_data.h"
#include "external/ge_common/api_error_codes.h"

namespace fe {
class PlatFormInfos;
} // namespace fe

namespace gert {
struct Dim3 {
  uint32_t x;
  uint32_t y;
  uint32_t z;

  Dim3(const uint32_t dim3_x, const uint32_t dim3_y = 1, const uint32_t dim3_z = 1) : x(dim3_x), y(dim3_y), z(dim3_z) {}
};

/**
 * tiling kernel的context
 */
class TilingContext : public ExtendedKernelContext {
 public:
  const void* GetCompileInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<void*>();
  }
  /**
   * 获取CompileInfo
   * @tparam T CompileInfo的类型
   * @return CompileInfo的指针
   */
  template <typename T>
  const T* GetCompileInfo() const {
    return reinterpret_cast<const T*>(GetCompileInfo());
  }
  /**
   * 获取输入shape，输入shape中包含了原始shape与运行时shape
   * @param index 输入index
   * @return 输入shape指针，index非法时返回空指针
   */
  const StorageShape* GetInputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetInputsNum()) {
      return nullptr;
    }
    return GetInputPointer<StorageShape>(index);
  }
  /**
   * 获取输入tensor
   * 若算子被配置为'tiling'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param index 输入index
   * @return 输入tensor指针，index非法时返回空指针
   */
  const Tensor* GetInputTensor(const size_t index) const {
    return GetInputPointer<Tensor>(index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor指针
   * 若算子被配置为'tiling'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return tensor指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Tensor* GetOptionalInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入Tensor指针
   * 若算子被配置为'tiling'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return Tensor指针，index或relative_index非法时，返回空指针
   */
  const Tensor* GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const StorageShape* GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor指针
   * 若算子被配置为'tiling'数据依赖，则返回的Tensor对象中保存了Host内存地址；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return Tensor指针，index非法时，返回空指针
   */
  const Tensor* GetRequiredInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const StorageShape* GetRequiredInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return shape指针，index或relative_index非法时，返回空指针
   */
  const StorageShape* GetDynamicInputShape(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, relative_index);
  }
  /**
   * 根据输出index，获取输出shape指针，shape中包含了原始shape与运行时shape
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回空指针
   */
  const StorageShape* GetOutputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }

    const size_t offset = compute_node_info->GetInputsNum();
    return GetInputPointer<StorageShape>(offset + index);
  }

  /*
   * outputs, tiling的outputs以如下顺序排列：
   * outputs[0]: tiling-key
   * outputs[1]: block-dim
   * outputs[2]: atomic-clean-flag
   * outputs[3]: tiling-data
   * outputs[4]: workspace sizes
   * outputs[5]: tiling condition
   * outputs[6]: schedule mode
   * outputs[7]: local memory size
   * outputs[8]: aicpu block-dim
   */
  enum TilingOutputIndex : uint32_t {
    kOutputTilingKey = 0,
    kOutputBlockDim = 1, // 接口即将废弃；优先选择 kOutputSimdNumBlocks
    kOutputSimdNumBlocks = 1,
    kOutputAtomicCleanFlag = 2,
    kOutputTilingData = 3,
    kOutputWorkspace = 4,
    kOutputTilingCond = 5,
    kOutputScheduleMode = 6,
    kOutputLocalMemorySize = 7, // 接口即将废弃；优先选择 kOutputDynUBufSize
    kOutputDynUBufSize = 7,
    kOutputAicpuBlockDim = 8, // 接口即将废弃；优先选择 kOutputAicpuNumBlocks
    kOutputAicpuNumBlocks = 8,
    kOutputSimtBlockDim = 9,
    kOutputSimtGridDim = 10,
    // add new output definitions here
    kOutputNum
  };

  /*
   * outputs[0]: fallible tiling condition
   */
  enum FallibleTilingOutputIndex : uint32_t {
    kTilingStatus = TilingOutputIndex::kOutputNum,
    // add new output definitions here
    kFallibleOutputNum
  };

  /**
   * 设置tiling key
   * @param tiling_key tiling key
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetTilingKey(const uint64_t tiling_key) {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_key;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取tiling key
   * @return tiling key，获取失败时
   */
  uint64_t GetTilingKey() const {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return std::numeric_limits<uint64_t>::max();
    }
    return *p;
  }

  /**
   * 设置schedule_mode
   * @param schedule_mode schedule_mode
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetScheduleMode(const uint32_t schedule_mode) {
    const auto p = GetOutputPointer<uint32_t>(kOutputScheduleMode);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = schedule_mode;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取设置schedule_mode
   * @return 设置schedule_mode，获取失败时
   */
  uint32_t GetScheduleMode() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputScheduleMode);
    if (p == nullptr) {
      return 0U;
    }
    return *p;
  }

  /**
   * 设置block dim
   * @param block_dim block dim
   * @return 成功时返回ge::GRAPH_SUCCESS
   * @deprecated SetBlockDim 接口即将废弃，改用 SetSimdNumBlocks
   * 接口；当前同时保留两个接口，后续整改完成后，删除废弃接口
   */
  ge::graphStatus SetBlockDim(const uint32_t block_dim) {
    return SetSimdNumBlocks(block_dim);
  }

  /**
   * 设置 simd number of blocks
   * @param num_blocks number of blocks
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetSimdNumBlocks(const uint32_t num_blocks) {
    const auto p = GetOutputPointer<uint32_t>(kOutputSimdNumBlocks);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = num_blocks;
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 获取block dim
   * @return block dim
   * @deprecated GetBlockDim 接口即将废弃，改用 GetSimdNumBlocks
   * 接口；当前同时保留两个接口，后续整改完成后，删除废弃接口
   */
  uint32_t GetBlockDim() const {
    return GetSimdNumBlocks();
  }

  /**
   * 获取 simd number of blocks
   * @return number of blocks
   */
  uint32_t GetSimdNumBlocks() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputSimdNumBlocks);
    if (p == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    return *p;
  }

  /**
   * 设置aicpu block dim(融合算子使用)
   * @param block_dim block dim
   * @return 成功时返回ge::GRAPH_SUCCESS
   * @deprecated SetAicpuBlockDim 接口即将废弃，改用 SetAicpuNumBlocks
   * 接口；当前同时保留两个接口，后续整改完成后，删除废弃接口
   */
  ge::graphStatus SetAicpuBlockDim(uint32_t block_dim) {
    return SetAicpuNumBlocks(block_dim);
  }

  /**
   * 设置 aicpu number of blocks(融合算子使用)
   * @param num_blocks number of blocks
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetAicpuNumBlocks(uint32_t num_blocks) {
    const auto p = GetOutputPointer<uint32_t>(kOutputAicpuNumBlocks);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = num_blocks;
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 获取aicpu block dim(融合算子使用)
   * @return block dim
   * @deprecated GetAicpuBlockDim 接口即将废弃，改用 GetAicpuNumBlocks
   * 接口；当前同时保留两个接口，后续整改完成后，删除废弃接口
   */
  uint32_t GetAicpuBlockDim() const {
    return GetAicpuNumBlocks();
  }

  /**
   * 获取 aicpu number of blocks(融合算子使用)
   * @return number of blocks
   */
  uint32_t GetAicpuNumBlocks() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputAicpuNumBlocks);
    if (p == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    return *p;
  }

  /**
   * 设置tiling cond
   * @param tiling_cond tiling condition
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetTilingCond(int32_t tiling_cond) {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_cond;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取tiling cond
   * @return tiling cond:有效的tiling_cond大于等于0，若该值无效返回-1
   */
  int32_t GetTilingCond() const {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return -1;
    }
    return *p;
  }
  /**
   * 设置是否需要atomic clean
   * @param atomic true/false代表是否需要做atomic clean
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetNeedAtomic(const bool atomic) {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = atomic;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取是否需要atomic clean
   * @return true/false
   */
  bool NeedAtomic() const {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return false;
    }
    return *p;
  }
  /**
   * 获取有类型的tiling data指针
   * @tparam T tiling data类型，sizeof(T)不可以大于编译结果中指定的最大tiling data长度
   * @return tiling data指针，失败时返回空指针
   */
  template <typename T>
  auto GetTilingData() -> T* {
    auto tiling_data = GetRawTilingData();
    if (tiling_data == nullptr) {
      return nullptr;
    }
    if (tiling_data->GetCapacity() < sizeof(T)) {
      return nullptr;
    }
    tiling_data->SetDataSize(sizeof(T));
    return static_cast<T*>(tiling_data->GetData());
  }
  /**
   * 获取无类型的tiling data码流
   * @return tiling data指针，失败时返回空指针
   */
  TilingData* GetRawTilingData() {
    const auto p = GetOutputPointer<TilingData*>(kOutputTilingData);
    return p == nullptr ? nullptr : *p;
  }
  /**
   * 获取workspace sizes指针
   * @param workspace_count workspace的个数，传入的workspace个数不可以超过编译时指定的最大workspace个数
   * @return workspace sizes指针
   */
  size_t* GetWorkspaceSizes(const size_t workspace_count) {
    const auto workspace = GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return nullptr;
    }
    if (workspace->SetSize(workspace_count) != ge::SUCCESS) {
      return nullptr;
    }
    return workspace->MutableData();
  }
  /**
   * 获取 workspace 个数
   * @return workspace 个数
   */
  size_t GetWorkspaceNum() const {
    const auto workspace = GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return 0U;
    }
    return workspace->GetSize();
  }
  /**
   * 获取 fe::PlatFormInfos 指针
   * @return fe::PlatFormInfos 指针
   */
  fe::PlatFormInfos* GetPlatformInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index + 1U);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<fe::PlatFormInfos*>();
  }

  /**
   * 获取 确定性计算变量
   * @return int32 变量
   */
  int32_t GetDeterministic() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    // 此处按照tiling内存排布，将确定性计算的字段添加在
    // inputshape outputshape compileinfo platform tiling_func之后
    const auto av = GetInput(index + 3U);
    if (av == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    return av->GetValue<int32_t>();
  }

  /**
   * 获取 确定性计算级别变量
   * @return int32 变量
   */
  int32_t GetDeterministicLevel() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    // 此处按照tiling内存排布，将确定性计算级别的字段添加在
    // inputshape outputshape compileinfo platform tiling_func deterministic之后
    const auto av = GetInput(index + 4U);
    if (av == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    return av->GetValue<int32_t>();
  }

  /**
   * 查询当前上下文是否启用PCIE Through
   * @return 任意一个输入为pcie，该接口返回true
   */
  bool GetPcieThroughFlag() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return false;
    }
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    // 此处按照tiling内存排布，将PCIE Through的字段添加在
    // inputshape outputshape compileinfo platform tiling_func deterministic pcie_through之后
    const auto av = GetInput(index + 5U);
    if (av == nullptr) {
      return false;
    }
    return av->GetValue<bool>();
  }

  /**
   * 设置 local memory size, 默认值为0
   * @param local_memory_size
   * @return 成功返回ge::GRAPH_SUCCESS
   * @deprecated SetLocalMemorySize 接口即将废弃，改用 SetDynUBufSize
   * 接口；当前同时保留两个接口，后续整改完成后，删除废弃接口
   */
  ge::graphStatus SetLocalMemorySize(const uint32_t local_memory_size) {
    return SetDynUBufSize(local_memory_size);
  }
  /**
   * 设置 dynamic unified buffer size, 默认值为0
   * @param dyn_ubuf_size
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetDynUBufSize(const uint32_t dyn_ubuf_size) {
    const auto p = GetOutputPointer<uint32_t>(kOutputDynUBufSize);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = dyn_ubuf_size;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取 local memory size，默认值为0
   * @return local memory size， 失败返回 std::numeric_limits<uint32_t>::max()
   * @deprecated GetLocalMemorySize 接口即将废弃，改用 GetDynUBufSize
   * 接口；当前同时保留两个接口，后续整改完成后，删除废弃接口
   */
  uint32_t GetLocalMemorySize() const {
    return GetDynUBufSize();
  }
  /**
   * 获取 dynamic unified buffer size，默认值为0
   * @return dynamic unified buffer size， 失败返回 std::numeric_limits<uint32_t>::max()
   */
  uint32_t GetDynUBufSize() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputDynUBufSize);
    if (p == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    return *p;
  }

  /**
   * 设置 SIMT Grid 维度
   * @param grid_dim
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetSimtGridDim(const Dim3& grid_dim) {
    const auto p = GetOutputPointer<Dim3>(kOutputSimtGridDim);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = grid_dim;
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 获取 SIMT Grid 维度
   * @return simt grid dim
   */
  const Dim3* GetSimtGridDim() const {
    return GetOutputPointer<Dim3>(kOutputSimtGridDim);
  }

  /**
   * 设置 SIMT Block 维度
   * @param block_dim
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetSimtBlockDim(const Dim3& block_dim) {
    const auto p = GetOutputPointer<Dim3>(kOutputSimtBlockDim);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = block_dim;
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 获取 SIMT Block 维度
   * @return simt block dim
   */
  const Dim3* GetSimtBlockDim() const {
    return GetOutputPointer<Dim3>(kOutputSimtBlockDim);
  }

  /**
   * 根据算子输入索引获取对应的输入是否携带非连续信息
   * 返回true仅表示可通过GetInputStride/GetInputOffset获取stride与offset，
   * 不代表输入实际是非连续tensor，输入可能为携带默认stride的连续tensor
   * @param index 输入index
   * @return 是否携带非连续信息，index非法或输入不携带非连续信息时返回false
   */
  bool InputIsView(const size_t index) const {
    if (GetInputViewTensor(index) == nullptr) {
      return false;
    }

    return true;
  }
  /**
   * 获取输入的stride
   * @index 输入index
   * @return 输入的stride，index非法或者tensor不携带非连续描述信息时返回空指针
   */
  const Stride* GetInputStride(const size_t index) const {
    const auto tensor = GetInputViewTensor(index);
    if (tensor == nullptr) {
      return nullptr;
    }

    return &(tensor->GetStride());
  }
  /**
   * 获取输入的offset
   * @index 输入index
   * @return 输入offset，index非法或者tensor不携带非连续描述信息时返回-1
   */
  int64_t GetInputOffset(const size_t index) const {
    const auto tensor = GetInputViewTensor(index);
    if (tensor == nullptr) {
      return -1;
    }

    return tensor->GetOffset();
  }
  /**
   * 根据算子输出索引获取对应的输出是否携带非连续信息
   * 返回true仅表示可通过GetOutputStride/GetOutputOffset获取stride与offset，
   * 不代表输出实际是非连续tensor，输出可能为携带默认stride的连续tensor
   * @param index 输出index
   * @return 是否携带非连续信息，index非法或输出不携带非连续信息时返回false
   */
  bool OutputIsView(const size_t index) const {
    if (GetOutputViewTensor(index) == nullptr) {
      return false;
    }

    return true;
  }
  /**
   * 获取输出的stride
   * @index 输出index
   * @return 输出的stride，index非法或者tensor不携带非连续描述信息时返回空指针
   */
  const Stride* GetOutputStride(const size_t index) const {
    const auto tensor = GetOutputViewTensor(index);
    if (tensor == nullptr) {
      return nullptr;
    }

    return &(tensor->GetStride());
  }
  /**
   * 获取输出的offset
   * @index 输出index
   * @return 输出offset，index非法或者tensor不携带非连续描述信息时返回-1
   */
  int64_t GetOutputOffset(const size_t index) const {
    const auto tensor = GetOutputViewTensor(index);
    if (tensor == nullptr) {
      return -1;
    }

    return tensor->GetOffset();
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor的view信息
   * 判断输入是否携带非连续描述信息
   * @param  ir_index IR原型定义中的index
   * @return 是否为非连续tensor，ir_index非法，或该INPUT没有实例化时，或该tensor未携带非连续描述信息时返回false
   */
  bool OptionalInputIsView(const size_t ir_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, 0);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return false;
    }

    return true;
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor的view信息
   * 获取输入的stride
   * @param  ir_index
   * IR原型定义中的index，ir_index非法，或该INPUT没有实例化，或者tensor不携带非连续描述信息时，返回空指针
   * @return 输入stride
   */
  const Stride* GetOptionalInputStride(const size_t ir_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, 0);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return nullptr;
    }

    return &(tensor->GetStride());
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor的view信息
   * 获取输入的offset
   * @param  ir_index IR原型定义中的index，ir_index非法，或该INPUT没有实例化，或者tensor不携带非连续描述信息时，返回-1
   * @return 输入offset
   */
  int64_t GetOptionalInputOffset(const size_t ir_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, 0);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return -1;
    }

    return tensor->GetOffset();
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入Tensor的view信息
   * 判断输入是否为非连续tensor
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return 是否为非连续tensor，ir_index或relative_index非法，或者tensor不携带非连续描述信息时，返回false
   */
  bool DynamicInputIsView(const size_t ir_index, const size_t relative_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, relative_index);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return false;
    }

    return true;
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入Tensor的view信息
   * 获取输入的stride
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return 输入stride，ir_index或relative_index非法，或者tensor不携带非连续描述信息时，返回空指针
   */
  const Stride* GetDynamicInputStride(const size_t ir_index, const size_t relative_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, relative_index);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return nullptr;
    }

    return &((tensor)->GetStride());
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入Tensor的view信息
   * 获取输入的offset
   * @param ir_index IR原型定义中的index
   * @param relative_index
   * 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return 输入offset，ir_index或relative_index非法，或者tensor不携带非连续描述信息时，返回-1
   */
  int64_t GetDynamicInputOffset(const size_t ir_index, const size_t relative_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, relative_index);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return -1;
    }

    return tensor->GetOffset();
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor的view信息
   * 判断输入是否为非连续tensor
   * @param ir_index IR原型定义中的index
   * @return 是否为非连续tensor，ir_index非法，或该INPUT没有实例化，或者tensor不携带非连续描述信息时，返回false
   */
  bool RequiredInputIsView(const size_t ir_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, 0);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return false;
    }

    return true;
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor的view信息
   * 获取输入的stride
   * @param ir_index IR原型定义中的index
   * @return 输入stride，ir_index非法，或该INPUT没有实例化，或者tensor不携带非连续描述信息时，返回空指针
   */
  const Stride* GetRequiredInputStride(const size_t ir_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, 0);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return nullptr;
    }

    return &(tensor->GetStride());
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor的view信息
   * 获取输入的offset
   * @param ir_index IR原型定义中的index
   * @return 输入offset，ir_index非法，或该INPUT没有实例化，或者tensor不携带非连续描述信息时，返回-1
   */
  int64_t GetRequiredInputOffset(const size_t ir_index) const {
    const auto tensor = GetDynamicInputPointer<TensorV2>(ir_index, 0);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return -1;
    }

    return tensor->GetOffset();
  }

 private:
  /**
   * 获取携带非连续描述信息的输入tensor
   * @param index 输入index
   * @return 携带非连续描述信息的输入tensor，index非法或者tensor不携带非连续描述信息时返回空指针
   */
  const TensorV2* GetInputViewTensor(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetInputsNum()) {
      return nullptr;
    }

    const auto tensor = GetInputPointer<TensorV2>(index);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return nullptr;
    }

    return tensor;
  }
  /**
   * 获取携带非连续描述信息的输出tensor
   * @param index 输出index
   * @return 携带非连续描述信息的输出tensor，index非法或者tensor不携带非连续描述信息时返回空指针
   */
  const TensorV2* GetOutputViewTensor(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }

    const size_t offset = compute_node_info->GetInputsNum();
    const auto tensor = GetInputPointer<TensorV2>(offset + index);
    if ((tensor == nullptr) || (tensor->GetVersion() == kTensorV1)) {
      return nullptr;
    }

    return tensor;
  }
};
static_assert(std::is_standard_layout<TilingContext>::value, "The class TilingContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
