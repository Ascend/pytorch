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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& l1_loss_backward_out_npu_nocheck(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
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

Tensor& l1_loss_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& grad_input) {
  Tensor gradOutputBroadcast = grad_output;
  Tensor targetBroadcast = target;
  if (grad_output.sizes() != self.sizes()) {
    gradOutputBroadcast = at::npu_broadcast(grad_output, self.sizes());
  }
  if (target.sizes() != self.sizes()) {
    targetBroadcast = at::npu_broadcast(target, self.sizes());
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

Tensor l1_loss_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor gradOutputBroadcast = grad_output;
  Tensor targetBroadcast = target;
  if (grad_output.sizes() != self.sizes()) {
    gradOutputBroadcast = at::npu_broadcast(grad_output, self.sizes());
  }
  if (target.sizes() != self.sizes()) {
    targetBroadcast = at::npu_broadcast(target, self.sizes());
  }
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  OpPipeWithApplyOut pipe;
  return pipe.ApplyOutputSameAs(self)
    .Func([&gradOutputBroadcast, &self, &targetBroadcast, &reductionStr](Tensor& result) {
      l1_loss_backward_out_npu_nocheck(result, gradOutputBroadcast, self, targetBroadcast, reductionStr);
    })
    .Call();
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("l1_loss_backward", TORCH_FN(l1_loss_backward_npu));
  m.impl("l1_loss_backward.grad_input", TORCH_FN(l1_loss_backward_out_npu));
}
} // namespace native
} // namespace at