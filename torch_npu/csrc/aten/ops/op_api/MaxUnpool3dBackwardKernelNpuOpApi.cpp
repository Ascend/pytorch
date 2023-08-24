// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::max_unpool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnMaxUnpool3dBackward,
                  NPUNativeFunctions::max_unpool3d_backward_out(grad_output, self, indices, output_size,
                                                                stride, padding, grad_input));
  OpPreparation::check_tensor({grad_output, self, indices}, grad_input, self);
  EXEC_NPU_CMD(aclnnMaxUnpool3dBackward, grad_output, self, indices, output_size, stride, padding, grad_input);
  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::max_unpool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  DO_COMPATIBILITY(aclnnMaxUnpool3dBackward,
                   NPUNativeFunctions::max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding));
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  EXEC_NPU_CMD(aclnnMaxUnpool3dBackward, grad_output, self, indices, output_size, stride, padding, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu

