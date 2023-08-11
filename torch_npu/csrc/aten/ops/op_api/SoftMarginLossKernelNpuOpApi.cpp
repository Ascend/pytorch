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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::soft_margin_loss_out(const at::Tensor& self, const at::Tensor& target,
                                                          int64_t reduction, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSoftMarginLoss, NPUNativeFunctions::soft_margin_loss_out(self, target, reduction, result));
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = broadcast_ops_npu_output_size(self, target);
  }
  if (result.sizes() != output_size) {
    result.resize_(output_size);
  }
  OpPreparation::CheckOut({self, target}, result, result.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnSoftMarginLoss, self, target, reduction, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::soft_margin_loss(const at::Tensor& self, const at::Tensor& target,
                                                     int64_t reduction) {
  DO_COMPATIBILITY(aclnnSoftMarginLoss, NPUNativeFunctions::soft_margin_loss(self, target, reduction));
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = broadcast_ops_npu_output_size(self, target);
  }
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  EXEC_NPU_CMD(aclnnSoftMarginLoss, self, target, reduction, result);
  return result;
}

} // namespace native
} // namespace at_npu
