// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::binary_cross_entropy_with_logits_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction) {

  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  const at::Tensor& pos_weight = c10::value_or_else(pos_weight_opt, [] {return at::Tensor();});

  at::Tensor gradInput = OpPreparation::ApplyTensor(self);
  at::Tensor weightTensor;
  if (weight.defined()) {
    weightTensor = NpuUtils::format_contiguous(weight);
    weightTensor = (weightTensor.scalar_type() != self.scalar_type()) ?
        NPUNativeFunctions::npu_dtype_cast(weightTensor, self.scalar_type()) : weightTensor;
  } else {
    weightTensor = at::ones(self.sizes(), self.options());
  }
  
  at::Tensor posWeightTensor;
  if (pos_weight.defined()) {
    posWeightTensor = NpuUtils::format_contiguous(pos_weight);
    posWeightTensor = (posWeightTensor.scalar_type() != self.scalar_type()) ?
        NPUNativeFunctions::npu_dtype_cast(posWeightTensor, self.scalar_type()) : posWeightTensor;
  } else {
    posWeightTensor = at::ones(self.sizes(), self.options());
  }
 
  at::Tensor doutTensor = NPUNativeFunctions::npu_broadcast(grad_output, self.sizes());
  std::string reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpCommand cmd;
  cmd.Name("SigmoidCrossEntropyWithLogitsGradV2")
      .Input(self)
      .Input(target)
      .Input(doutTensor)
      .Input(weightTensor)
      .Input(posWeightTensor)
      .Output(gradInput)
      .Attr("reduction", reductionStr)
      .Run();

  return gradInput;
}
} // namespace native
} // namespace at_npu