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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor, at::Tensor> prelu_backward_out_npu_nocheck(
    at::Tensor& grad_input, 
    at::Tensor& grad_weight,
    const at::Tensor& grad_output, 
    const at::Tensor& self, 
    const at::Tensor& weight) {
  OpCommand cmd;
  cmd.Name("PReluGrad")
      .Input(grad_output)
      .Input(self)
      .Input(weight)
      .Output(grad_input)
      .Output(grad_weight)
      .Run();
  return tuple<at::Tensor, at::Tensor>(grad_input, grad_weight);
}

tuple<at::Tensor, at::Tensor> prelu_backward_out_npu(
    at::Tensor& grad_input, 
    at::Tensor& grad_weight,
    const at::Tensor& grad_output, 
    const at::Tensor& self, 
    const at::Tensor& weight) {
  prelu_backward_out_npu_nocheck(grad_input, grad_weight, grad_output, self, weight);
  return tuple<at::Tensor, at::Tensor>(grad_input, grad_weight);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_prelu_kernel_backward(
    const at::Tensor& grad_output, 
    const at::Tensor& self, 
    const at::Tensor& weight) {
  // construct the output Tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  at::Tensor grad_weight = OpPreparation::ApplyTensor(weight);
  // calculate the output result of the NPU
  prelu_backward_out_npu_nocheck(grad_input, grad_weight, grad_output, self, weight);
  return std::tie<at::Tensor, at::Tensor>(grad_input, grad_weight);
}
} // namespace native
} // namespace at_npu