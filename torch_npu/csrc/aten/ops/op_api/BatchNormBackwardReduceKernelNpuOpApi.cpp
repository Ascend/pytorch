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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> NPUNativeOpApiFunctions::batch_norm_backward_reduce(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight_opt,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  // set compatibility
  DO_COMPATIBILITY(
        aclnnBatchNormReduceBackward,
        NPUNativeFunctions::batch_norm_backward_reduce(grad_out, self, mean, invstd, weight_opt,
                                                       input_g, weight_g, bias_g));

  int64_t n_input = self.size(1);
  at::Tensor sum_dy;
  at::Tensor sum_dy_xmu;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  bool is_float16 = false;
  if (self.scalar_type() == mean.scalar_type() && self.scalar_type() == at::kHalf) {
    is_float16 = true;
  }

  auto out_dtype = (is_float16) ? at::kHalf : at::kFloat;

  // construct the output tensor of the NPU
  if (input_g) {
      sum_dy = OpPreparation::ApplyTensorWithoutFormat(mean.sizes(), mean.options().dtype(out_dtype));
      sum_dy_xmu = OpPreparation::ApplyTensorWithoutFormat(mean.sizes(), mean.options().dtype(out_dtype));
  }

  if (weight_g) {
      grad_weight = OpPreparation::ApplyTensorWithoutFormat({n_input}, invstd.options().dtype(out_dtype));
  }

  if (bias_g) {
      grad_bias = OpPreparation::ApplyTensorWithoutFormat({n_input}, grad_out.options().dtype(out_dtype));
  }

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnBatchNormReduceBackward,
               grad_out,
               self,
               mean,
               invstd,
               weight_opt,
               input_g,
               weight_g,
               bias_g,
               sum_dy,
               sum_dy_xmu,
               grad_weight,
               grad_bias);

  return std::make_tuple(sum_dy, sum_dy_xmu, grad_weight, grad_bias);
}

}  // namespace native
}  // namespace at_npu
