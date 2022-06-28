// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <c10/util/Exception.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCmdHelper.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu {
namespace native {

OpCommand& OpCommand::Name(const string &name) {
    IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(graphCmd.SetName(name);)
    aclCmd->SetName(name);
    return *this;
}

OpCommand& OpCommand::DynamicInputReg(
    DynamicInputRegFunc func,
    DyNumAndIndex num_and_index) {
  IF_GRAPH_MODE_THEN_RUN(
    graphCmd.AddDynamicInputRegFunc(func, num_and_index);)
return *this;
}

OpCommand& OpCommand::Expect(UnifiedResult unified_result) {
  commonType = unified_result.common_type;
  resultTypeDefined = unified_result.result_type_defined;
  commonShape = unified_result.common_shape;
  return *this;
}

OpCommand& OpCommand::Input() {
  IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
      graphCmd.AddInput();
  )
  return AddNoneTensor();
}

OpCommand& OpCommand::Input(
    const at::Tensor &input,
    const string &descName,
    const c10::optional<aclFormat> &sensitive_format,
    const string &realData) {
  IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
      auto contiguous_input = Contiguous(input);
      if (commonType.has_value() &&
          commonType.value() != contiguous_input.scalar_type()) {
        contiguous_input = NPUNativeFunctions::npu_dtype_cast(contiguous_input, commonType.value());
      }
      graphCmd.AddInput(contiguous_input, descName, realData, sensitive_format);
  )
  return AddTensorInput(
      Contiguous(input), c10::ScalarType::Undefined, descName, realData);
}

OpCommand& OpCommand::InputWithoutContiguous(
    const at::Tensor &input,
    const string &descName,
    const string &realData) {
  IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
      graphCmd.AddInput(input, descName, realData);
  )
  if (input.storage_offset() != 0) {
    TORCH_WARN_ONCE(
        "[Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy",
        input.storage_offset());
  }
  return AddTensorInput(const_cast<at::Tensor &>(input));
}

OpCommand& OpCommand::Input(const c10::IntArrayRef &dimListRef, at::ScalarType toType) {
  IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
      graphCmd.AddInput(dimListRef, toType);
  )
  at::Tensor &cpuTensor = CreateHostTensor((void *) dimListRef.data(),
                                           dimListRef.size(),
                                           c10::TensorOptions(at::kCPU).dtype(at::kLong),
                                           toType);
  return AddHostTensorInput(cpuTensor);
}

OpCommand& OpCommand::Input(const c10::Scalar &input, const at::ScalarType type,
    CompileType compileType) {
  IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
      auto true_type = commonType.has_value() ? commonType.value() : type;
      graphCmd.AddInput(input, true_type, compileType);
  )
  const auto &scalarTensor = CreateScalarTensor(input, type);
  return AddHostTensorInput(scalarTensor, compileType);
}

OpCommand& OpCommand::Inputs(const at::TensorList &inputs)
{
  for (auto &input : inputs)
  {
    this->Input(input);
  }
  return *this;
}

OpCommand& OpCommand::Output(
    at::Tensor &output,
    const string &descName,
    const c10::optional<aclFormat> &sensitive_format,
    const string &realType) {
  IF_GRAPH_MODE_THEN_RUN_WITH_RET_THIS(
      if (sensitive_format.has_value() &&
          FormatHelper::GetBaseFormat(output) != sensitive_format.value()) {
        output = NPUNativeFunctions::npu_format_cast(output, sensitive_format.value());
      }
      graphCmd.AddOutput(output, descName, realType, sensitive_format);
      if (!resultTypeDefined && commonType.has_value() &&
          output.scalar_type() != commonType.value()) {
        output = NPUNativeFunctions::npu_dtype_cast(output, commonType.value());
      } 
  )
  outputTensor.emplace_back(output);
  return AddOutput(output, realType);
}

void OpCommand::Run() {
  IF_GRAPH_MODE_THEN_RUN(return;)
  if (c10_npu::option::OptionsManager::CheckQueueEnable() && !sync) {
    ExecuteParas execParams;
    aclCmd->ExportParams(execParams);
    c10_npu::queue::QueueParas params(c10_npu::queue::COMPILE_AND_EXECUTE, sizeof(ExecuteParas), &execParams);
    c10_npu::enCurrentNPUStream(&params);
    aclCmd->releaseSource(false);
  } else {
    aclCmd->Run(sync, sync_index, outputTensor);
    aclCmd->releaseSource();
  } 
  aclCmds->Pop();
}

OpCommand& OpCommand::Sync(c10::SmallVector<int64_t, N> &index) {
  sync_index = index;
  if (!index.empty()) {
    sync = true;
  }
  return *this;
}

OpCommand& OpCommand::AddTensorInput(at::Tensor &tensor,
                                     at::ScalarType forceScaleType,
                                     const string &descName,
                                     const string &realData) {
  std::tuple < aclTensorDesc * , aclDataBuffer *> res;
  if (commonType.has_value() && commonType.value() != tensor.scalar_type()) {
    tensor = NPUNativeFunctions::npu_dtype_cast(tensor, commonType.value());
  }
  // 针对dim=0的场景，绝对不会有输入为uint16的情况，因为这个是TBE引入的，TBE没有dim=0的情况
  if (tensor.dim() == 0) {
    if (at_npu::key::isDeviceTensor(tensor)) {
      res = OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(tensor, descName);
    } else {
      res = OpCmdHelper::CovertTensorWithZeroDimToAclInput(tensor, forceScaleType);
    }
  } else {
    res = OpCmdHelper::CovertTensorToAclInput(tensor, descName, realData);
  }
  aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
  return *this;
}

OpCommand& OpCommand::AddHostTensorInput(const at::Tensor &tensor, CompileType compileType) {
  std::tuple < aclTensorDesc *, aclDataBuffer *> res;
  res = OpCmdHelper::CovertHostTensorToAclInput(tensor, tensor.scalar_type(), compileType);
  aclCmd->AddInput(std::get<0>(res), std::get<1>(res), tensor);
  return *this;
}

OpCommand& OpCommand::AddNoneTensor() {
  AclTensorDescMaker desc;
  auto aclDesc = desc.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED).Get();
  AclTensorBufferMaker buffer(nullptr, 0);
  aclCmd->AddInput(aclDesc, buffer.Get());
  return *this;
}

OpCommand& OpCommand::AddOutput(at::Tensor &output, const string &realType) {
  if (resultTypeDefined == false && commonType.has_value() && commonType.value() != output.scalar_type()) {
    output = NPUNativeFunctions::npu_dtype_cast(output, commonType.value());
  }
  auto res = OpCmdHelper::CovertToAclOutput(output, realType);
  aclCmd->AddOutput(std::get<0>(res), std::get<1>(res));
  return *this;
}

// 由于format_contiguous会生成新Tensor，为了保证其在生命周期内有效，故而放到对象中存储
// 同下，CopyScalarToDevice也有同样问题
at::Tensor& OpCommand::Contiguous(const at::Tensor &input) {
  storage.emplace_back(std::move(NpuUtils::format_contiguous_add_copy_optimize(input)));
  return storage.back();
}

at::Tensor& OpCommand::CreateHostTensor(
    void *data, size_t size,
    const c10::TensorOptions &options,
    at::ScalarType toType) {
  auto cpuTensor = at::empty(size, options);
  std::memcpy(cpuTensor.data_ptr(), data, sizeof(int64_t) * cpuTensor.numel());
  if (toType != at::kLong) {
    cpuTensor = cpuTensor.to(toType);
  }

  storage.emplace_back(std::move(cpuTensor));
  return storage.back();
}

at::Tensor& OpCommand::CreateScalarTensor(const c10::Scalar &scalar, at::ScalarType type) {
  if (commonType.has_value()) {
    type = commonType.value();
  }
  storage.emplace_back(scalar_to_tensor(scalar).to(type));
  return storage.back();
}

} // namespace native
} // namespace at_npu