// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor& NPUNativeOpApiFunctions::elu_out(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale,
                                             const at::Scalar& input_scale, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnElu, NPUNativeFunctions::elu_out(self, alpha, scale, input_scale, result));
  OpPreparation::CheckOut({self}, result, result, self.sizes());
  EXEC_NPU_CMD(aclnnElu, self, alpha, scale, input_scale, result);
  return result;
}

at::Tensor elu_npu_impl_op_api(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale) {
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  EXEC_NPU_CMD(aclnnElu, self, alpha, scale, input_scale, result);
  return result;
}

at::Tensor elu_backward_npu_impl_op_api(const at::Tensor& grad_output, at::Scalar alpha, at::Scalar scale,
                                        at::Scalar input_scale, const at::Tensor& output) {
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(grad_output);
  bool is_result = true;
  EXEC_NPU_CMD(aclnnEluBackward, grad_output, alpha, scale, input_scale, is_result, output, result);
  return result;
}

class NPUEluOpApiFunction: public torch::autograd::Function<NPUEluOpApiFunction> {
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
    at::Tensor result = elu_npu_impl_op_api(self, alpha, scale, input_scale);
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
    auto grad_input = elu_backward_npu_impl_op_api(
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

at::Tensor NPUNativeOpApiFunctions::elu(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale,
                                        const at::Scalar& input_scale) {
  DO_COMPATIBILITY(aclnnElu, NPUNativeFunctions::elu(self, alpha, scale, input_scale));
  return NPUEluOpApiFunction::apply(self, alpha, scale, input_scale);
}

at::Tensor& NPUNativeOpApiFunctions::elu_(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale,
                                          const at::Scalar& input_scale) {
  DO_COMPATIBILITY(aclnnElu, NPUNativeFunctions::elu_(self, alpha, scale, input_scale));
  auto result = NPUEluOpApiFunction::apply(self, alpha, scale, input_scale);
  self.copy_(result);
  return self;
}

} // namespace native
} // namespace at_npu
