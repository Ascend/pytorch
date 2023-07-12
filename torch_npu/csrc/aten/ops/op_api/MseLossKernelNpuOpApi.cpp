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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::mse_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMseLossOut, NPUNativeFunctions::mse_loss_out(self, target, reduction, result));
  auto output_size = mse_loss_npu_output_size(self, target, reduction);
  OpPreparation::CheckOut({self, target}, result, result.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnMseLossOut, self, target, reduction, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::mse_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  DO_COMPATIBILITY(aclnnMseLoss, NPUNativeFunctions::mse_loss(self, target, reduction));
  c10::SmallVector<int64_t, SIZE> output_size;
  if (reduction == at::Reduction::None) {
    output_size = broadcast_ops_npu_output_size(self, target);
  }
  at::ScalarType high_type = at::native::result_type(self, target);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(high_type));
  EXEC_NPU_CMD(aclnnMseLoss, self, target, reduction, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
