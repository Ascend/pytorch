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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor& l1_loss_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    string reductionStr) {
  OpCommand cmd;
  cmd.Name("LpLoss")
      .Input(self)
      .Input(target)
      .Attr("reduction", reductionStr)
      .Attr("p", (int64_t)1)
      .Output(result)
      .Run();
  return result;
}

at::Tensor npu_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  }
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  string reductionStr = CalcuOpUtil::GetReductionStr(reduction);

  l1_loss_out_npu_nocheck(result, self, target, reductionStr);
  return result;
}

at::Tensor& l1_loss_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    string reductionStr) {
  OpCommand cmd;
  cmd.Name("L1LossGrad")
      .Input(grad_output)
      .Input(self)
      .Input(target)
      .Attr("reduction", reductionStr)
      .Output(grad_input)
      .Run();
  return grad_input;
}

at::Tensor npu_l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::Tensor gradOutputBroadcast = grad_output;
  at::Tensor targetBroadcast = target;
  if (grad_output.sizes() != self.sizes()) {
    gradOutputBroadcast = NPUNativeFunctions::npu_broadcast(grad_output, self.sizes());
  }
  if (target.sizes() != self.sizes()) {
    targetBroadcast = NPUNativeFunctions::npu_broadcast(target, self.sizes());
  }
  string reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpPipeWithApplyOut pipe;
  return pipe.ApplyOutputSameAs(self)
    .Func([&gradOutputBroadcast, &self, &targetBroadcast, &reductionStr](at::Tensor& result) {
      l1_loss_backward_out_npu_nocheck(result, gradOutputBroadcast, self, targetBroadcast, reductionStr);
    })
    .Call();
}

class NPUL1LossFunction : public torch::autograd::Function<NPUL1LossFunction> {
public:
  static at::Tensor forward(
      AutogradContext *ctx,
      const at::Tensor& self,
      const at::Tensor& target,
      int64_t reduction) {
    at::AutoNonVariableTypeMode g;
    auto result = npu_l1_loss(self, target, reduction);
    ctx->save_for_backward({self, target});
    ctx->saved_data["reduction"] = reduction;
    return result;
  }

  static tensor_list backward(
      AutogradContext *ctx,
      tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto reduction = ctx->saved_data["reduction"].toInt();
    auto grad_input = npu_l1_loss_backward(grad_outputs[0], saved[0], saved[1], reduction);
    tensor_list output = {grad_input, -grad_input, at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  return NPUL1LossFunction::apply(self, target, reduction);
}

} // namespace native
} // namespace at_npu
