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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::batch_norm_stats(const at::Tensor& self, double eps) {
  DO_COMPATIBILITY(aclnnBatchNormStats, NPUNativeFunctions::batch_norm_stats(self, eps));
  TORCH_CHECK(self.ndimension() >= 2, "Expected 2D+ Tensor, but got tensor with ", self.ndimension(), " Dimension");
  at::Tensor mean = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  at::Tensor invstd = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);

  EXEC_NPU_CMD(aclnnBatchNormStats, self, eps, mean, invstd);
  return std::tie(mean, invstd);
}
}  // namespace native
}  // namespace at_npu