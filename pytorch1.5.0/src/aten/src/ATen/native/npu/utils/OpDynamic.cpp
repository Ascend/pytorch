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

#include "OpDynamic.h"
#include "c10/npu/OptionsManager.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/frame/OpCmdHelper.h"
#include "ATen/native/npu/utils/DynamicShapeUtil.h"
#include "ATen/native/npu/frame/OpDynamicParamMaker.h"

namespace at {
namespace native {
namespace npu {

OpDynamicCommand::OpDynamicCommand() {
  aclDynamicCmd = OpDynamicCommandImpl::GetInstance();
  aclDynamicCmd->UpdateDynamicParam();
}

OpDynamicCommand::~OpDynamicCommand() {}

// OpDynamicCommand Part
OpDynamicCommand& OpDynamicCommand::InputPair(const Tensor& npu_input, const Tensor& cpu_input) {
  return AddTensorInput(Contiguous(npu_input), ScalarType::Undefined, "", "", cpu_input);
}

OpDynamicCommand& OpDynamicCommand::Inputs(const TensorList& inputs) {
  for (auto& input : inputs) {
    this->Input(input);
  }
  return *this;
}

OpDynamicCommand& OpDynamicCommand::DynamicExpect(UnifiedResult unified_result) {
  commonType = unified_result.common_type;
  resultTypeDefined = unified_result.result_type_defined;
  commonShape = unified_result.common_shape;
  return *this;
}

OpDynamicCommand& OpDynamicCommand::DynamicOutput(Tensor& output, string realType, shapeStrage strage, bool isDimZeroToOne){
  if (resultTypeDefined == false && commonType.has_value()) {
    output = output.npu_dtype_cast(commonType.value());
  }
  
  const Tensor* tensor = &output;
  
  // dynmaic compile
  aclTensorDesc* compileRes = OpDynamicCmdHelper::CovertToAclOutputDynamicCompileDesc(tensor, realType, strage, isDimZeroToOne);
  aclDynamicCmd->AddDynamicOutputDesc(compileRes);

  // dynamic Run
  auto runRes = OpCmdHelper::CovertToAclOutput(tensor, realType);
  aclDynamicCmd->AddDynamicOutput(std::get<0>(runRes), std::get<1>(runRes), std::get<2>(runRes), std::get<3>(runRes));
  return *this;
}

OpDynamicCommand& OpDynamicCommand::DynamicInput(
    SmallVector<int64_t, N>& dimList,
    ScalarType originType,
    ScalarType toType,
    string descName,
    bool isConst,
    shapeStrage strage) {
  
  Tensor cpuTensor = from_blob((void*)dimList.data(), {dimList.size()}, originType).to(toType);
  Tensor npuTensor = CopyHostToDevice(cpuTensor);
  
  if (isConst == true) {
    return DynamicInput(npuTensor, descName, "", cpuTensor, strage);
  } else {
    c10::optional<Tensor> cpu_tensor = c10::nullopt;
    return DynamicInput(npuTensor, descName, "", cpu_tensor, strage);
  }
}

OpDynamicCommand& OpDynamicCommand::DynamicInput(const Tensor& npu_input,
    string descName,
    string realData,
    c10::optional<Tensor> cpu_tensor,
    shapeStrage strage) {
  std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat> runRes;
  aclTensorDesc* compileRes = nullptr;
  Tensor npuTensor = Contiguous(npu_input);
  
  std::string dynamicKey;
  if (commonType.has_value()) {
    npuTensor = npuTensor.npu_dtype_cast(commonType.value());
  }

  // 针对dim=0的场景，绝对不会有输入为uint16的情况，因为这个是TBE引入的，TBE没有dim=0的情况
  if (npuTensor.dim() == 0) {
    std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat, aclTensorDesc*> dimZeroRes;
    if (npuTensor.is_npu()) {
      dimZeroRes = OpDynamicCmdHelper::CovertNPUTensorWithZeroDimToDynamicAclInput(npuTensor, descName);
    } else {
      dimZeroRes = OpDynamicCmdHelper::CovertTensorWithZeroDimToDynamicAclInput(npuTensor, ScalarType::Undefined);
    }

    aclDynamicCmd->AddDynamicInput(std::get<0>(dimZeroRes), 
        std::get<1>(dimZeroRes), 
        std::get<2>(dimZeroRes), 
        std::get<3>(dimZeroRes));
    aclDynamicCmd->AddDynamicCompileInputDesc(std::get<4>(dimZeroRes));
    aclDynamicCmd->AddDynamicKey("-1,_-1,;");

  } else {
    if (cpu_tensor.has_value() && cpu_tensor.value().defined()) {
      runRes = OpCmdHelper::CovertTensorToAclInput(npuTensor, cpu_tensor, descName, "");
      compileRes = OpDynamicCmdHelper::CovertToAclInputConstDynamicCompileDesc(npuTensor, cpu_tensor, dynamicKey, descName, "", strage);
    } else {
      runRes = OpCmdHelper::CovertTensorToAclInput(npuTensor, cpu_tensor, descName, realData);
      compileRes = OpDynamicCmdHelper::CovertToAclInputDynamicCompileDesc(npuTensor, cpu_tensor, dynamicKey, descName, realData, strage);
    }

    aclDynamicCmd->AddDynamicInput(std::get<0>(runRes), std::get<1>(runRes), std::get<2>(runRes), std::get<3>(runRes));
    aclDynamicCmd->AddDynamicCompileInputDesc(compileRes);
    aclDynamicCmd->AddDynamicKey(dynamicKey);
  }
  return *this;
}

OpDynamicCommand& OpDynamicCommand::DynamicName(string name) {
  aclDynamicCmd->SetDynamicName(name);
  return *this;
}

void OpDynamicCommand::DynamicOpRun(){
  if (c10::npu::OptionsManager::CheckQueueEnable()) {
    ExecuteParas execParams;
    aclCmd->ExportParams(execParams);
    aclDynamicCmd->ExportDynamicParams(execParams);
    QueueParas params(COMPILE_AND_EXECUTE, sizeof(ExecuteParas), &execParams);
    c10::npu::enCurrentNPUStream(&params);
    aclCmd->releaseSource(false);
    aclDynamicCmd->ReleaseDynamicSource(false);
  } else if (c10::npu::OptionsManager::CheckDynamicEnable()) {
    ExecuteParas runParams;
    auto stream = c10::npu::getCurrentNPUStream();
    aclCmd->ExportParams(runParams);
    aclDynamicCmd->ExportDynamicParams(runParams);
    DynamicRun(runParams, stream);
    runParams.DynamicRelease();
    runParams.Release();
    aclCmd->releaseSource(false);
    aclDynamicCmd->ReleaseDynamicSource(false);
  } else {}

  aclCmds->Pop();
}

} // namespace npu
} // namespace native
} // namespace at