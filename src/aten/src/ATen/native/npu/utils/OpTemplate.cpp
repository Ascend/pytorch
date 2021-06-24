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

#include "OpTemplate.h"
#include "ATen/native/GlobalStep.h"
#include "ATen/native/npu/frame/OpCmdHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/OpParamMaker.h"

namespace at {
namespace native {
namespace npu {

// TransDataOpCommand Part
TransDataOpCommand& TransDataOpCommand::InputAndOutput(const Tensor& input, const Tensor& output) {
  if (input.defined() == false) {
    AT_NPU_CHECK(ACL_ERROR_INVALID_PARAM);
  }
  return AddInputAndOutput(input, output);
}

TransDataOpCommand& TransDataOpCommand::AddInputAndOutput(const Tensor& input, const Tensor& output) {

  std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat> in;
  std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat> out;

   if (!c10::npu::OptionsManager::CheckDynamicEnable() && check_fuzz_enable()) {
    in = OpCmdHelper::CovertTensorToAclInput(input, c10::nullopt, "", "");
    out = OpCmdHelper::CovertTensorToAclInput(output, c10::nullopt, "", "");
  } else {
    in = OpCmdHelper::CovertTransDataTensorToAcl(input);
    out = OpCmdHelper::CovertTransDataTensorToAcl((output));
  }

  aclCmd->AddInput(
      std::get<0>(in), std::get<1>(in), std::get<2>(in), std::get<3>(in));
  aclCmd->AddOutput(
      std::get<0>(out), std::get<1>(out), std::get<2>(out), std::get<3>(out));
  return *this;
}

// OpCommand Part
OpCommand& OpCommand::InputPair(const Tensor& npu_input, const Tensor& cpu_input) {
  return AddTensorInput(Contiguous(npu_input), ScalarType::Undefined, "", "", cpu_input);
}

OpCommand& OpCommand::Inputs(const TensorList& inputs) {
  for (auto& input : inputs) {
    this->Input(input);
  }
  return *this;
}

OpCommand& OpCommand::InputWithFunc(const FUNC_TYPE& func) {
  auto res = func();
  if (std::get<0>(res)) {
    return *this;
  }
  return AddTensorInput(std::get<1>(res), ScalarType::Undefined, "", "");
}

} // namespace npu
} // namespace native
} // namespace at