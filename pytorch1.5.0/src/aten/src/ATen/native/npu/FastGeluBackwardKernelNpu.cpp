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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;


namespace {
Tensor& fast_gelu_backward_npu_nocheck(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& self) {
  // constructs the input and output NPUTensorDesc
  OpCommand cmd;
  cmd.Name("FastGeluGrad")
    .Input(grad)
    .Input(self)
    .Output(grad_input)
    .Run();

  return grad_input;
}

}

Tensor fast_gelu_backward_npu(
    const Tensor& grad, 
    const Tensor& self) {
  // calculate the output size
  // Tensor outputTensor = self;
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor grad_input = at::empty_with_format(
        outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  
  // calculate the output result of the NPU
  fast_gelu_backward_npu_nocheck(grad_input, grad, self);
  
  return grad_input;
}

} // namespace native
} // namespace at
