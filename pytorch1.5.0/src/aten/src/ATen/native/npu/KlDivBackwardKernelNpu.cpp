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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor kl_div_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto outputSize = input_same_output_size(self);
  Tensor grad_input = OpPreparation::ApplyTensor(outputSize, self.options(),self);
  
  string reductionStr;
  if (reduction == Reduction::Mean) {
    reductionStr = "batchmean";
  } else if (reduction == Reduction::Sum) {
    reductionStr = "sum";
  } else if (reduction == Reduction::None) {
    reductionStr = "none";
  }

  OpCommand cmd;
  cmd.Name("KlDivLossGrad")
      .Input(grad_output)
      .Input(self)
      .Input(target)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Attr("log_target", false)
      .Run();

  if (reduction == Reduction::Mean) {
    auto inputShape = self.sizes();
    int batchSquareSize = prod_intlist(inputShape) / inputShape[0];
    grad_input.div_(batchSquareSize);
  }

  return grad_input;
}

} // namespace native
} // namespace at