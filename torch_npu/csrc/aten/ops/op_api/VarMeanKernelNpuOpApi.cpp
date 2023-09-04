// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return NPUNativeOpApiFunctions::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::var_mean(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dims,
    c10::optional<int64_t> correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnVarMean, NPUNativeFunctions::var_mean(self, dims, correction, keepdim));
  c10::SmallVector<int64_t, SIZE> real_dim = {};
  if (dims.has_value()) {
    real_dim = array_to_small_vector(dims.value());
  }
  auto output_size = reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto real_correction = correction.has_value() ? correction.value() : 1;
  auto var = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());
  auto mean = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());

  EXEC_NPU_CMD(aclnnVarMean, self, dims, real_correction, keepdim, mean, var);
  return tuple<at::Tensor, at::Tensor>(mean, var);
}

} // namespace native
} // namespace at_npu
