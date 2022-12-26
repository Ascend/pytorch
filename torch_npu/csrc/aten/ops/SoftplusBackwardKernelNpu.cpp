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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& softplus_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold,
    const at::Tensor& output) {
  OpCommand cmd;
  cmd.Name("SoftplusV2Grad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Attr("beta", beta)
      .Attr("threshold", threshold)
      .Run();

    return grad_input;
}

at::Tensor& NPUNativeFunctions::softplus_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    const at::Tensor& output,
    at::Tensor& grad_input) {
  OpPreparation::CheckOut(
      {self},
      grad_input,
      self);
  return softplus_backward_out_nocheck(grad_input, grad_output, self, beta, threshold, output);
}

at::Tensor NPUNativeFunctions::softplus_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold,
    const at::Tensor& output) {
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize, self.options(), self);
  softplus_backward_out_nocheck(result, grad_output, self, beta, threshold, output);
  return result;
}

} // namespace native
} // namespace at_npu