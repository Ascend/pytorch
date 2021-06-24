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

#ifndef __NATIVE_NPU_UTILS_OP_DYNAMIC__
#define __NATIVE_NPU_UTILS_OP_DYNAMIC__

#include <functional>
#include "ATen/native/npu/frame/OpDynamicParamMaker.h"
#include "ATen/native/npu/frame/OpDynamicCmdHelper.h"
#include "ATen/native/npu/frame/OpCommandBase.h"

namespace at {
namespace native {
namespace npu {

class OpDynamicCommand : public OpCommandBase<OpDynamicCommand> {
public:
  OpDynamicCommand();
  ~OpDynamicCommand();
  OpDynamicCommand& Inputs(const TensorList& inputs);
  OpDynamicCommand& InputPair(const Tensor& npu_input, const Tensor& cpu_input);
  OpDynamicCommand& DynamicName(string name);
  OpDynamicCommand& DynamicOutput(Tensor& output, 
    string realType = "",
    shapeStrage strage = FIXED_NONE,
    bool isDimZeroToOne = true);

  OpDynamicCommand& DynamicInput(const Tensor& npu_input, 
    string descName = "",
    string realData = "",
    c10::optional<Tensor> cpu_tensor = c10::nullopt,
    shapeStrage strage = FIXED_NONE);
  
  OpDynamicCommand& DynamicInput(SmallVector<int64_t, N>& dimList,
    ScalarType originType,
    ScalarType toType,
    string descName = "",
    bool isConst = false,
    shapeStrage strage = FIXED_NONE);

  template <typename dataType>
  OpDynamicCommand& DynamicAttr(string name, dataType value) {
    aclDynamicCmd->AddDynamicAttr(name, value);
    return *this;
  }
  OpDynamicCommand& DynamicExpect(UnifiedResult unified_result);
  void DynamicOpRun();

private:
  OpDynamicCommand& AddInputAndOutput(const Tensor& input, const Tensor& output);
  c10::optional<ScalarType> commonType = c10::nullopt;
  c10::optional<IntArrayRef> commonShape = c10::nullopt;
  OpDynamicCommandImpl* aclDynamicCmd = nullptr; // owned
  bool resultTypeDefined = false;
}; // class OpDynamicCommand
} // namespace npu
} // namespace native
} // namespace at
#endif