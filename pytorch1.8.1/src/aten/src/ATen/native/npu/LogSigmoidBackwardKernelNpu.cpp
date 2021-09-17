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

Tensor& log_sigmoid_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer,
    Tensor& grad_input) {
  OpPreparation::CheckMemory({grad_output, self, buffer}, {grad_input});
  OpCommand cmd;
  cmd.Name("LogSigmoidGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Run();
  return grad_input;
}

Tensor log_sigmoid_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  // construct the output tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  // calculate the output result of the NPU
  log_sigmoid_backward_out_npu(grad_output, self, buffer, grad_input);

  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("log_sigmoid_backward", TORCH_FN(log_sigmoid_backward_npu));
  m.impl("log_sigmoid_backward.grad_input", TORCH_FN(log_sigmoid_backward_out_npu));
}
} // namespace native
} // namespace at