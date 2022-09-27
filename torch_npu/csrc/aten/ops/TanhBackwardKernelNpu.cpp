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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& tanh_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("TanhGrad")
    .Input(self)
    .Input(grad_output)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::tanh_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut({grad_output, self}, result, self);
  tanh_backward_out_npu_nocheck(result, grad_output, self);
  return result;
}

at::Tensor NPUNativeFunctions::tanh_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  tanh_backward_out_npu_nocheck(result, grad_output, self);

  return result;
}

} // namespace native
} // namespace at_npu
