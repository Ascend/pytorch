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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
namespace {
at::Tensor fast_gelu_npu_nocheck(at::Tensor& result, const at::Tensor& self) {

    OpCommand cmd;
    cmd.Name("FastGelu")
        .Input(self)
        .Output(result)
        .Run();

    return result;
}

} // namespace

namespace {
at::Tensor& fast_gelu_backward_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad,
    const at::Tensor& self) {
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

class NPUFastGeluFunction : public torch::autograd::Function<NPUFastGeluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self) {
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self});
    at::Tensor result = OpPreparation::ApplyTensor(self);
    return fast_gelu_npu_nocheck(result, self);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];

    at::Tensor result = NPUNativeFunctions::fast_gelu_backward(grad_outputs[0], input);
    tensor_list output = {result};
    return output;
  }
};

at::Tensor NPUNativeFunctions::fast_gelu_backward(
    const at::Tensor& grad, 
    const at::Tensor& self) {
  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  fast_gelu_backward_npu_nocheck(grad_input, grad, self);
  
  return grad_input;
}

at::Tensor NPUNativeFunctions::fast_gelu(const at::Tensor& self) {
    return NPUFastGeluFunction::apply(self);
}

} // namespace native
} // namespace at_npu