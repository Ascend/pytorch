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
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor& elu_out_nocheck(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, at::Tensor& result) {
  float alphaValue = CalcuOpUtil::GetScalarFloatValue(alpha);
  float scaleValue = CalcuOpUtil::GetScalarFloatValue(scale);
  float inputScaleValue = CalcuOpUtil::GetScalarFloatValue(input_scale);
  OpCommand cmd;
  cmd.Name("Elu")
     .Input(self)
     .Output(result)
     .Attr("alpha", alphaValue)
     .Attr("scale", scaleValue)
     .Attr("input_scale", inputScaleValue)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::elu_out(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor checkResult = elu_out_nocheck(self, alpha, scale, input_scale, contiguousResult);
    NpuUtils::format_fresh_view(result, checkResult);
  } else {
    elu_out_nocheck(self, alpha, scale, input_scale, result);
  }
  return result;
}

at::Tensor elu_npu_impl(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  elu_out_nocheck(self, alpha, scale, input_scale, result);
  return result;
}

at::Tensor& elu_backward_out_npu(at::Tensor& grad_input, const at::Tensor& grad_output, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, const at::Tensor& output) {
    float value = CalcuOpUtil::GetScalarFloatValue(alpha);
    OpCommand cmd;
    cmd.Name("EluGradV2")
       .Input(grad_output)
       .Input(output)
       .Output(grad_input)
       .Attr("alpha", value)
       .Run();
    return grad_input;
}
at::Tensor elu_backward_npu_impl(const at::Tensor& grad_output, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, const at::Tensor& output) {
    at::Tensor result = OpPreparation::ApplyTensor(grad_output);
    elu_backward_out_npu(result, grad_output, alpha, scale, input_scale, output);
    return result;
}

class NPUEluFunction: public torch::autograd::Function<NPUEluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self, 
      at::Scalar alpha, 
      at::Scalar scale, 
      at::Scalar input_scale) {
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["input_scale"] = input_scale;
    at::AutoNonVariableTypeMode g;
    at::Tensor result = elu_npu_impl(self, alpha, scale, input_scale);
    ctx->save_for_backward({result});
    return result;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto alpha = ctx->saved_data["alpha"].toScalar();
    auto scale = ctx->saved_data["scale"].toScalar();
    auto input_scale = ctx->saved_data["input_scale"].toScalar();
    auto saved = ctx->get_saved_variables();
    auto result = saved[0];
    auto grad_input = elu_backward_npu_impl(
        grad_outputs[0], 
        alpha,
        scale,
        input_scale, 
        result);
    tensor_list output = {grad_input,
                          at::Tensor(),
                          at::Tensor(),
                          at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::elu(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale) {
  return NPUEluFunction::apply(self, alpha, scale, input_scale);
}

at::Tensor& NPUNativeFunctions::elu_(at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale) {
  auto result = NPUEluFunction::apply(self, alpha, scale, input_scale);
  self.copy_(result);
  return self;
}

} // namespace native
} // namespace at_npu
