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

at::Tensor& NPUNativeFunctions::hardtanh_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar min_val,
    at::Scalar max_val,
    at::Tensor& grad_input) {
  OpPreparation::CheckMemory({grad_output, self}, {grad_input});
  OpCommand cmd;
  cmd.Name("HardtanhGrad")
      .Input(self)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("max_val", max_val)
      .Attr("min_val", min_val)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::hardtanh_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar min_val,
    at::Scalar max_val) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  hardtanh_backward_out(grad_output, self, min_val, max_val, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
