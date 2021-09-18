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

namespace {
Tensor& hardsigmoid_backward_nocheck(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self) {
  OpCommand cmd;
  cmd.Name("HardSigmoidGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Run();

  return grad_input;
}
} // namespace

Tensor hardsigmoid_backward_npu(
    const Tensor& grad_output,
    const Tensor& self) {
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  // calculate the output result of the NPU
  hardsigmoid_backward_nocheck(grad_input, grad_output, self);

  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("hardsigmoid_backward", TORCH_FN(hardsigmoid_backward_npu));
}
} // namespace native
} // namespace at
