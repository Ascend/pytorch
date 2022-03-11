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

at::Tensor& hardshrink_backward_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar lambd) {
  OpCommand cmd;
  cmd.Name("HardShrinkGrad")
      .Input(grad_output)
      .Input(self)
      .Attr("lambd", lambd)
      .Output(grad_input)
      .Run();

  return grad_input;
}

at::Tensor NPUNativeFunctions::hardshrink_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar lambd) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  hardshrink_backward_nocheck(grad_input, grad_output, self, lambd);

  return grad_input;
}

} // namespace native
} // namespace at_npu