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

#include <ATen/Tensor.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeOpApiFunctions::native_batch_norm(
    const at::Tensor& self, const c10::optional<at::Tensor>& weight_opt, const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt, const c10::optional<at::Tensor>& running_var_opt, bool train,
    double momentum, double eps) {
  DO_COMPATIBILITY(aclnnBatchNorm, NPUNativeFunctions::native_batch_norm(self, weight_opt, bias_opt, running_mean_opt,
                                                                         running_var_opt, train, momentum, eps));
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self.sizes(), self.options(), self);
  at::Tensor save_mean;
  at::Tensor save_invstd;
  if (train) {
    save_mean = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
    save_invstd = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  } else {
    save_mean = at::empty({0}, self.options());
    save_invstd = at::empty({0}, self.options());
  }

  EXEC_NPU_CMD(aclnnBatchNorm, self, weight_opt, bias_opt, running_mean_opt, running_var_opt, train, momentum, eps,
               result, save_mean, save_invstd);

  return std::tie(result, save_mean, save_invstd);
}
}  // namespace native
}  // namespace at_npu
