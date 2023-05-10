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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;

at::Tensor& selu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Selu")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor selu_npu_impl(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  selu_out_npu_nocheck(result, self);
  return result;
}

at::Tensor& selu_backward_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("SeluGrad")
      .Input(grad_output)
      .Input(result)
      .Output(grad_input)
      .Run();
  return grad_input;
}

at::Tensor selu_backward_npu_impl(const at::Tensor& grad_output, const at::Tensor& result) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  selu_backward_npu_nocheck(grad_input, grad_output, result);
  return grad_input;
}

class NPUSeluFunction : public torch::autograd::Function<NPUSeluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self) {
    at::AutoNonVariableTypeMode g;
    at::Tensor result = selu_npu_impl(self);
    ctx->save_for_backward({result});
    return result;
  }

  static tensor_list backward(AutogradContext *ctx,
      tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto result = saved[0];
    auto grad_input = selu_backward_npu_impl(grad_outputs[0], result);
    tensor_list output = {grad_input};
    return output;
  }
};

at::Tensor NPUNativeFunctions::selu(const at::Tensor& self) {
  return NPUSeluFunction::apply(self);
}

at::Tensor& NPUNativeFunctions::selu_(at::Tensor& self) {
  at::Tensor result = NPUSeluFunction::apply(self);
  self.copy_(result);
  return self;
}

} // namespace native
} // namespace at_npu
