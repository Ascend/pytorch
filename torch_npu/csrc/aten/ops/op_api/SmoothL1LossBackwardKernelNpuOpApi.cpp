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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::smooth_l1_loss_backward_out(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta,
    at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnSmoothL1LossBackward,
                   NPUNativeFunctions::smooth_l1_loss_backward_out(grad_out, self, target, reduction, beta, grad_input));
  auto output_size = smooth_l1_loss_backward_npu_output_size(self, target, grad_out);
  OpPreparation::CheckOut({grad_out, self, target}, grad_input, grad_input.scalar_type(), output_size);
  float sigma = static_cast<float>(beta);
  EXEC_NPU_CMD(aclnnSmoothL1LossBackward, grad_out, self, target, reduction, sigma, grad_input);
  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::smooth_l1_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  DO_COMPATIBILITY(aclnnSmoothL1LossBackward,
                   NPUNativeFunctions::smooth_l1_loss_backward(grad_out, self, target, reduction, beta));
  auto output_size = smooth_l1_loss_backward_npu_output_size(self, target, grad_out);
  at::Tensor grad_input = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  float sigma = static_cast<float>(beta);
  EXEC_NPU_CMD(aclnnSmoothL1LossBackward, grad_out, self, target, reduction, sigma, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
