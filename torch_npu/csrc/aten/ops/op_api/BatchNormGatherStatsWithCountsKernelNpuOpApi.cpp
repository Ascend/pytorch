// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::batch_norm_gather_stats_with_counts(
    const at::Tensor& self, const at::Tensor& mean, const at::Tensor& invstd,
    const c10::optional<at::Tensor>& running_mean_opt, const c10::optional<at::Tensor>& running_var_opt,
    double momentum, double eps, const at::Tensor& counts) {
  DO_COMPATIBILITY(aclnnBatchNormGatherStatsWithCounts,
                   NPUNativeFunctions::batch_norm_gather_stats_with_counts(self, mean, invstd, running_mean_opt,
                                                                           running_var_opt, momentum, eps, counts));
  at::Tensor mean_all = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  at::Tensor invstd_all = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  EXEC_NPU_CMD(aclnnBatchNormGatherStatsWithCounts, self, mean, invstd, running_mean_opt, running_var_opt, momentum,
               eps, counts, mean_all, invstd_all);
  return std::make_tuple(mean_all, invstd_all);
}

}  // namespace native
}  // namespace at_npu
