// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;

at::Tensor& celu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar& alpha) {
  OpCommand cmd;
  cmd.Name("CeluV2")
      .Input(self)
      .Output(result)
      .Attr("alpha", alpha)
      .Run();

  return result;
}

at::Tensor celu_npu_impl(const at::Tensor& self, at::Scalar& alpha) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  celu_out_npu_nocheck(result, self, alpha);
  return result;
}

at::Tensor& celu_backward_out_npu(at::Tensor& grad_input, const at::Tensor& grad_output,
    at::Scalar& alpha, const at::Tensor& output) {
  OpCommand cmd;
  cmd.Name("EluGradV2")
      .Input(grad_output)
      .Input(output)
      .Output(grad_input)
      .Attr("alpha", alpha)
      .Run();
  return grad_input;
}

at::Tensor celu_backward_npu_impl(const at::Tensor& grad_output, at::Scalar& alpha, const at::Tensor& output) {
  at::Tensor result = OpPreparation::ApplyTensor(grad_output);
  celu_backward_out_npu(result, grad_output, alpha, output);
  return result;
}

class NPUCeluFunction : public torch::autograd::Function<NPUCeluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
                            const at::Tensor& self,
                            at::Scalar alpha) {
    ctx->saved_data["alpha"] = alpha;
    at::AutoNonVariableTypeMode g;
    at::Tensor result = celu_npu_impl(self, alpha);
    ctx->save_for_backward({result});
    return result;
  }

  static tensor_list backward(AutogradContext *ctx,
                              tensor_list grad_outputs) {
    auto alpha = ctx->saved_data["alpha"].toScalar();
    auto saved = ctx->get_saved_variables();
    auto result = saved[0];
    auto grad_input = celu_backward_npu_impl(
        grad_outputs[0],
        alpha,
        result);
    tensor_list output = {grad_input, at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::celu(const at::Tensor& self, const at::Scalar& alpha) {
  return NPUCeluFunction::apply(self, alpha);
}

at::Tensor& NPUNativeFunctions::celu_(at::Tensor& self, const at::Scalar& alpha) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = NPUCeluFunction::apply(contiguousSelf, alpha);
    NpuUtils::format_fresh_view(self, result);
  } else {
    auto result = NPUCeluFunction::apply(self, alpha);
    self.copy_(result);
  }
  return self;
}

} // namespace native
} // namespace at_npu
