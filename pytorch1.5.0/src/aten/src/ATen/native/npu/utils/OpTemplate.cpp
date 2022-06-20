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
#include "ATen/native/npu/interface/EnvVariables.h"
#include "ATen/native/npu/frame/OpCmdHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/OpParamMaker.h"

namespace at {
namespace native {
namespace npu {

// OpCommand Part
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
  IF_GRAPH_MODE_THEN_RUN(
      graphCmd.AddInput(std::get<1>(res), "", "");
      return *this;
  )
  return AddTensorInput(std::get<1>(res), ScalarType::Undefined, "", "");
}

} // namespace npu
} // namespace native
} // namespace at