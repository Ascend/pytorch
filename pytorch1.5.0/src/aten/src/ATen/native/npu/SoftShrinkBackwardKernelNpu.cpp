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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& softshrink_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd) {
  float lambd_value = CalcuOpUtil::get_scalar_float_value(lambd);
  OpCommand cmd;
  cmd.Name("SoftShrinkGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Attr("lambd", lambd_value)
      .Run();
  return grad_input;
}

Tensor softshrink_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd) {
  // construct the output tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  softshrink_backward_out_npu(
      grad_input, grad_output, self, lambd);

  return grad_input;
}

}
}