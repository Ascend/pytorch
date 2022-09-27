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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::log_sigmoid_forward_out(
    const at::Tensor& self,
    at::Tensor& output,
    at::Tensor& buffer) {
  OpCommand cmd;
  cmd.Name("LogSigmoid")
      .Input(self)
      .Output(output)
      .Run();
  return std::tie(output, buffer);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::log_sigmoid_forward(const at::Tensor& self) {
  at::Tensor output = OpPreparation::ApplyTensor(self);
  at::Tensor buffer = OpPreparation::ApplyTensorWithSizes({0}, self.options());
  // calculate the output result of the NPU
  log_sigmoid_forward_out(self, output, buffer);
  return tuple<at::Tensor, at::Tensor>(output, buffer);
}

at::Tensor& NPUNativeFunctions::log_sigmoid_out(const at::Tensor& self, at::Tensor& result) {
  at::Tensor buffer = OpPreparation::ApplyTensorWithSizes({0}, self.options());
  return std::get<0>(at::log_sigmoid_forward_out(result, buffer, self));
}

at::Tensor NPUNativeFunctions::log_sigmoid(const at::Tensor& self) {
  return std::get<0>(at::log_sigmoid_forward(self));
}

} // namespace native
} // namespace at_npu