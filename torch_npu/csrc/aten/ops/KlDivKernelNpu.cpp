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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor npu_kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target) {
  at::Tensor result =
      reduction == at::Reduction::None ?
      OpPreparation::ApplyTensor(self) :
      OpPreparation::ApplyTensor({}, self.options(), self);
  string reductionStr;
  if (reduction == at::Reduction::Mean) {
    reductionStr = "batchmean";
  } else if (reduction == at::Reduction::Sum) {
    reductionStr = "sum";
  } else if (reduction == at::Reduction::None) {
    reductionStr = "none";
  }
  OpCommand cmd;
  cmd.Name("KLDiv")
      .Input(self)
      .Input(target)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Attr("log_target", log_target)
      .Run();
  if (reduction == at::Reduction::Mean) {
    auto inputShape = self.sizes();
    int batchSquareSize = c10::multiply_integers(inputShape) / inputShape[0];
    result.div_(batchSquareSize);
  }
  return result;
}

at::Tensor npu_kl_div_backward(
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
  if (reduction == at::Reduction::Mean) {
    auto inputShape = self.sizes();
    int batchSquareSize = c10::multiply_integers(inputShape) / inputShape[0];
    grad_input.div_(batchSquareSize);
  }
  return grad_input;
}

class NPUKlDivFunction : public torch::autograd::Function<NPUKlDivFunction> {
public:
  static at::Tensor forward(
      AutogradContext *ctx,
      const at::Tensor& self,
      const at::Tensor& target,
      int64_t reduction,
      bool log_target) {
    at::AutoNonVariableTypeMode g;
    auto result = npu_kl_div(self, target, reduction, log_target);
    ctx->save_for_backward({self, target});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["log_target"] = log_target;
    return result;
  }

  static tensor_list backward(
      AutogradContext *ctx,
      tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto reduction = ctx->saved_data["reduction"].toInt();
    auto log_target = ctx->saved_data["log_target"].toBool();
    auto grad_input = npu_kl_div_backward(grad_outputs[0], saved[0], saved[1], reduction, log_target);
    tensor_list output = {grad_input, at::Tensor(), at::Tensor(), at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target) {
    return NPUKlDivFunction::apply(self, target, reduction, log_target);
}

} // namespace native
} // namespace at_npu
