// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

at::Tensor& NPUNativeOpApiFunctions::batch_norm_elemt_out(const at::Tensor& self,
                                                          const c10::optional<at::Tensor>& weight_opt,
                                                          const c10::optional<at::Tensor>& bias_opt,
                                                          const at::Tensor& mean, const at::Tensor& invstd, double eps,
                                                          at::Tensor& result) {
  DO_COMPATIBILITY(aclnnBatchNormElemt,
                   NPUNativeFunctions::batch_norm_elemt_out(self, weight_opt, bias_opt, mean, invstd, eps, result));
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] { return at::Tensor(); });
  OpPreparation::CheckOut({self, weight, bias, mean, invstd}, result, self);
  EXEC_NPU_CMD(aclnnBatchNormElemt, self, weight, bias, mean, invstd, eps, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::batch_norm_elemt(const at::Tensor& self, const c10::optional<at::Tensor>& weight,
                                                     const c10::optional<at::Tensor>& bias, const at::Tensor& mean,
                                                     const at::Tensor& invstd, double eps) {
  DO_COMPATIBILITY(aclnnBatchNormElemt, NPUNativeFunctions::batch_norm_elemt(self, weight, bias, mean, invstd, eps));
  at::Tensor result = OpPreparation::ApplyTensor(self);
  EXEC_NPU_CMD(aclnnBatchNormElemt, self, weight, bias, mean, invstd, eps, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
