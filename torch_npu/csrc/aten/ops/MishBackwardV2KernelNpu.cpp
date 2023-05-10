// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

at::Tensor& mish_backward_out_npu_nocheck(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& grad_input) {
  OpCommand cmd;
  cmd.Name("MishGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)    
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::mish_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  mish_backward_out_npu_nocheck(grad_output, self, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
