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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include <c10/core/ScalarType.h>

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::l1_loss_backward_out(const at::Tensor& grad_output,
                                                          const at::Tensor& self,
                                                          const at::Tensor& target,
                                                          int64_t reduction,
                                                          at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnL1LossBackward,
                   NPUNativeFunctions::l1_loss_backward_out(grad_output, self, target, reduction, grad_input));
  // check if grad_input on NPU
  TORCH_CHECK(at_npu::key::isDeviceTensor(grad_input), "grad_input with device ", grad_input.device(),
              " doesn't match the desired device NPU");
  at::IntArrayRef output_size1;
  at::IntArrayRef output_size2;
  output_size1 = broadcast_ops_npu_output_size(self, target);
  output_size2 = broadcast_ops_npu_output_size(output_size1, grad_output.sizes());
  if (grad_input.sizes() != output_size2) {
    grad_input.resize_(output_size2);
  }
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnL1LossBackward, grad_output, self, target, reduction, grad_input);
  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::l1_loss_backward(const at::Tensor& grad_output,
                                                     const at::Tensor& self,
                                                     const at::Tensor& target,
                                                     int64_t reduction) {
  DO_COMPATIBILITY(aclnnL1LossBackward, NPUNativeFunctions::l1_loss_backward(grad_output, self, target, reduction));
  // construct the output tensor of NPU
  at::IntArrayRef output_size1;
  at::IntArrayRef output_size2;
  output_size1 = broadcast_ops_npu_output_size(self, target);
  output_size2 = broadcast_ops_npu_output_size(output_size1, grad_output.sizes());
  // dtype promotion
  auto promote1 = at::native::result_type(target, self);
  auto grad_input_dtype = promoteTypes(grad_output.scalar_type(), promote1);
  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(output_size2, self.options().dtype(grad_input_dtype), self);
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnL1LossBackward, grad_output, self, target, reduction, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
