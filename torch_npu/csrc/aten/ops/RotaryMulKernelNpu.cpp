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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& rotary_mul_nocheck(
    at::Tensor& y,
    const at::Tensor& x,
    const at::Tensor& r1,
    const at::Tensor& r2) {
  OpCommand cmd;
  cmd.Name("RotaryMul")
      .Input(x)
      .Input(r1)
      .Input(r2)
      .Output(y)
      .Run();
  return y;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> rotary_mul_backward_nocheck(
    at::Tensor& dx,
    at::Tensor& dr1,
    at::Tensor& dr2,
    const at::Tensor& x,
    const at::Tensor& r1,
    const at::Tensor& r2,
    const at::Tensor& dy) {
  OpCommand cmd;
  cmd.Name("RotaryMulGrad")
      .Input(x)
      .Input(r1)
      .Input(r2)
      .Input(dy)
      .Output(dx)
      .Output(dr1)
      .Output(dr2)
      .Run();
  return std::tie(dx, dr1, dr2);
}

at::Tensor NPUNativeFunctions::npu_rotary_mul(
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  rotary_mul_nocheck(result, self, r1, r2);
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_rotary_mul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2) {
  at::Tensor dx = OpPreparation::ApplyTensor(self);
  at::Tensor dr1 = OpPreparation::ApplyTensor(r1);
  at::Tensor dr2 = OpPreparation::ApplyTensor(r2);
  rotary_mul_backward_nocheck(dx, dr1, dr2, self, r1, r2, grad);
  return std::tie(dx, dr1, dr2);
}

} // namespace native
} // namespace at_npu
