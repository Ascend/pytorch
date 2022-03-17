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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpTemplate.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::kl_div_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target) {
  auto outputSize = input_same_output_size(self);
  at::Tensor grad_input = OpPreparation::ApplyTensor(outputSize, self.options(), self);
  string reductionStr;
  if (reduction == at::Reduction::Mean) {
    reductionStr = "batchmean";
  } else if (reduction == at::Reduction::Sum) {
    reductionStr = "sum";
  } else if (reduction == at::Reduction::None) {
    reductionStr = "none";
  }
  OpCommand cmd;
  cmd.Name("KlDivLossGrad")
      .Input(grad_output)
      .Input(self)
      .Input(target)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Attr("log_target", log_target)
      .Run();
  return grad_input;
}
} // namespace native
} // namespace at_npu
