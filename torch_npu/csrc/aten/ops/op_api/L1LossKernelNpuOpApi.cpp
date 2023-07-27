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

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::l1_loss_out(const at::Tensor& self,
                                                 const at::Tensor& target,
                                                 int64_t reduction,
                                                 at::Tensor& result) {
  DO_COMPATIBILITY(aclnnL1Loss, NPUNativeFunctions::l1_loss_out(self, target, reduction, result));
  // check if result on NPU
  TORCH_CHECK(at_npu::key::isDeviceTensor(result), "result with device ", result.device(),
              " doesn't match the desired device NPU");
  // 1. When reduction = 'none', shape of result must be the same as self.
  // 2. When reduction != 'none', result must be a 0-dimensional tensor.
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = broadcast_ops_npu_output_size(self, target);
  }
  // Shape of result must be the same as self, dtype has no limitation.
  if (result.sizes() != output_size) {
    result.resize_(output_size);
  }
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnL1Loss, self, target, reduction, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::l1_loss(const at::Tensor& self,
                                            const at::Tensor& target,
                                            int64_t reduction) {
  DO_COMPATIBILITY(aclnnL1Loss, NPUNativeFunctions::l1_loss(self, target, reduction));
  // construct the output tensor of NPU
  // 1. If reduction='none', the output size should be the same size as self.
  // 2. Otherwise pass {} to ApplyTensor.
  // 3. Dtype of output should be the same dtype as self.
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = broadcast_ops_npu_output_size(self, target);
  }
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnL1Loss, self, target, reduction, result);
  return result;
}

} // namespace native
} // namespace at_npu
