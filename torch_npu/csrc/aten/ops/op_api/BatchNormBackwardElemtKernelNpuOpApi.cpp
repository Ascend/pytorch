// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::batch_norm_backward_elemt(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight_opt,
    const at::Tensor& mean_dy,
    const at::Tensor& mean_dy_xmu,
    const at::Tensor& count) {
  DO_COMPATIBILITY(aclnnBatchNormElemtBackward, NPUNativeFunctions::batch_norm_backward_elemt(grad_out, input, mean,
                                                                                              invstd, weight_opt,
                                                                                              mean_dy, mean_dy_xmu,
                                                                                              count));

  const at::Tensor &weight = c10::value_or_else(weight_opt, [] { return at::Tensor(); });

  at::Tensor grad_input = OpPreparation::ApplyTensor(input);
  EXEC_NPU_CMD(aclnnBatchNormElemtBackward, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, grad_input);
  return grad_input;
}
} // namespace native
} // namespace at_npu
