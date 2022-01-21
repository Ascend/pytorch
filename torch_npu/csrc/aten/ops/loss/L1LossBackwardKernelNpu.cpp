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

at::Tensor& NPUNativeFunctions::l1_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  at::Tensor gradOutputBroadcast = grad_output;
  at::Tensor targetBroadcast = target;
  if (grad_output.sizes() != self.sizes()) {
    gradOutputBroadcast = NPUNativeFunctions::npu_broadcast(grad_output, self.sizes());
  }
  if (target.sizes() != self.sizes()) {
    targetBroadcast = NPUNativeFunctions::npu_broadcast(target, self.sizes());
  }

  OpPreparation::CheckOut(
      {grad_output, self, target},
      grad_input,
      self);
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  return l1_loss_backward_out_npu_nocheck(
      grad_input,
      gradOutputBroadcast,
      self,
      targetBroadcast,
      reductionStr);
}

at::Tensor NPUNativeFunctions::l1_loss_backward(
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
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  OpPipeWithApplyOut pipe;
  return pipe.ApplyOutputSameAs(self)
    .Func([&gradOutputBroadcast, &self, &targetBroadcast, &reductionStr](at::Tensor& result) {
      l1_loss_backward_out_npu_nocheck(result, gradOutputBroadcast, self, targetBroadcast, reductionStr);
    })
    .Call();
}

} // namespace native
} // namespace at_npu