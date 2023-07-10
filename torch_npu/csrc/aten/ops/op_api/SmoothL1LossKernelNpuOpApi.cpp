// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor &NPUNativeOpApiFunctions::smooth_l1_loss_out(const at::Tensor &self, const at::Tensor &target,
                                                        int64_t reduction, double beta, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnSmoothL1Loss,
                   NPUNativeFunctions::smooth_l1_loss_out(self, target, reduction, beta, result));
  auto outputSize = smooth_l1_loss_npu_output_size(self, target, reduction);
  OpPreparation::CheckOut({self, target}, result, result.scalar_type(), outputSize);
  OpPreparation::CheckMemory({self, target}, {result});
  float sigma = static_cast<float>(beta);
  EXEC_NPU_CMD(aclnnSmoothL1Loss, self, target, reduction, sigma, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::smooth_l1_loss(const at::Tensor &self, const at::Tensor &target, int64_t reduction,
                                                   double beta) {
  DO_COMPATIBILITY(aclnnSmoothL1Loss,
                   NPUNativeFunctions::smooth_l1_loss(self, target, reduction, beta));
  auto outputSize = smooth_l1_loss_npu_output_size(self, target, reduction);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  float sigma = static_cast<float>(beta);
  EXEC_NPU_CMD(aclnnSmoothL1Loss, self, target, reduction, sigma, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
