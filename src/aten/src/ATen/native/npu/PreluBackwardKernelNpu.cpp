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
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor, Tensor> prelu_backward_out_npu(
    Tensor& grad_input, 
    Tensor& grad_weight,
    const Tensor& grad_output, 
    const Tensor& self, 
    const Tensor& weight) {
  OpCommand cmd;
  cmd.Name("PReluGrad")
      .Input(grad_output)
      .Input(self)
      .Input(weight)
      .Output(grad_input)
      .Output(grad_weight)
      .Run();

  return tuple<Tensor, Tensor>(grad_input, grad_weight);
}

tuple<Tensor, Tensor> prelu_backward_npu(
    const Tensor& grad_output, 
    const Tensor& self, 
    const Tensor& weight) {
  // calculate the output size
  auto outputSizes1 = input_same_output_size(self);
  auto outputSizes2 = input_same_output_size(weight);  
  // construct the output tensor of the NPU
  Tensor grad_input = at::empty_with_format(
    outputSizes1, self.options(), CalcuOpUtil::get_tensor_npu_format(self));  
  Tensor grad_weight = at::empty_with_format(
    outputSizes2, weight.options(), CalcuOpUtil::get_tensor_npu_format(weight));
  // calculate the output result of the NPU
  prelu_backward_out_npu(grad_input, grad_weight, grad_output, self, weight);
  return std::tie<Tensor, Tensor>(grad_input, grad_weight);
}

} // namespace native
} // namespace at