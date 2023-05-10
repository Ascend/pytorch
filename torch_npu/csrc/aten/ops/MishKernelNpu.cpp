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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor mish_npu(const at::Tensor& self) {
  at::Tensor result =  OpPreparation::ApplyTensor(self);
  OpCommand cmd;
  cmd.Name("Mish")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::npu_mish_backward(const at::Tensor& grad, const at::Tensor& input) {
  at::Tensor result =  OpPreparation::ApplyTensor(input);
  OpCommand cmd;
  cmd.Name("MishGrad")
      .Input(grad)
      .Input(input)
      .Output(result)
      .Run();

  return result;
}

class NPUMishFunction : public torch::autograd::Function<NPUMishFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self) {
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self});
    return mish_npu(self);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];

    at::Tensor result = NPUNativeFunctions::npu_mish_backward(grad_outputs[0], input);
    tensor_list output = {result};
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_mish(const at::Tensor& self) {
  return NPUMishFunction::apply(self);
}

} // namespace native
} // namespace at_npu