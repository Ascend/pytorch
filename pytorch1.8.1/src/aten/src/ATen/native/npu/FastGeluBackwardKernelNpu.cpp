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
#include <torch/csrc/autograd/custom_function.h>

namespace at {
namespace native {
using namespace at::native::npu;
using namespace torch::autograd;

namespace {
Tensor& fast_gelu_backward_npu_nocheck(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& self) {
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
  auto outputSize = input_same_output_size(self);

  Tensor grad_input = OpPreparation::ApplyTensor(self, outputSize);
  fast_gelu_backward_npu_nocheck(grad_input, grad, self);

  return grad_input;
}

class NPUFastGeluFunction : public torch::autograd::Function<NPUFastGeluFunction> {
public:
  static Tensor forward(AutogradContext *ctx,
    const Tensor& self) {
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self});
    return at::fast_gelu(self);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];
    Tensor result = fast_gelu_backward_npu(grad_outputs[0], self);
    tensor_list output = {result};
    return output;
  }
};

Tensor fast_gelu_autograd(const Tensor& self) {
  return NPUFastGeluFunction::apply(self);
}

TORCH_LIBRARY_IMPL(aten, AutogradNPU, m) {
  m.impl("fast_gelu", fast_gelu_autograd);
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("fast_gelu_backward", TORCH_FN(fast_gelu_backward_npu));
}

} // namespace native
} // namespace at
