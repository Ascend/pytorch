// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& silu_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("SiluGrad")
    .Input(grad_output)
    .Input(self)
    .Output(grad_input)
    .Run();

  return grad_input;
}

at::Tensor& NPUNativeFunctions::silu_backward_out(const at::Tensor& grad_output, const at::Tensor& self,
                                                  at::Tensor& grad_input) {
  OpPreparation::CheckOut({grad_output, self}, grad_input, grad_output);
  silu_backward_out_nocheck(grad_input, grad_output, self);
  return grad_input;
}

at::Tensor NPUNativeFunctions::silu_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  silu_backward_out_nocheck(grad_input, grad_output, self);
  return grad_input;
}

} // namespace native
} // namespace at_npu

