// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::soft_margin_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  string reductionStr(CalcuOpUtil::GetReductionStr(reduction));

  OpPreparation::CheckMemory({grad_output, input, target}, {grad_input});
  OpCommand cmd;
  cmd.Name("SoftMarginLossGrad")
      .Input(input)
      .Input(target)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::soft_margin_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(input);

  NPUNativeFunctions::soft_margin_loss_backward_out(
      grad_output, input, target, reduction, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu