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
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd,
    Tensor& grad_input) {
  OpPreparation::CheckMemory({grad_output, self}, {grad_input});
  OpCommand cmd;
  cmd.Name("SoftShrinkGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Attr("lambd", lambd)
      .Run();
  return grad_input;
}

Tensor softshrink_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd) {
  Tensor grad_input = OpPreparation::ApplyTensor(self);

  softshrink_backward_out_npu(
      grad_output, self, lambd, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("softshrink_backward", TORCH_FN(softshrink_backward_npu));
  m.impl("softshrink_backward.grad_input", TORCH_FN(softshrink_backward_out_npu));
}
}
}
