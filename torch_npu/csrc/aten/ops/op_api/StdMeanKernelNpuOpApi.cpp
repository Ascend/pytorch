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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

std::tuple <at::Tensor, at::Tensor> NPUNativeOpApiFunctions::std_mean(
    const at::Tensor & self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return NPUNativeOpApiFunctions::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple <at::Tensor, at::Tensor> NPUNativeOpApiFunctions::std_mean(
    const at::Tensor & self,
    c10::optional<at::IntArrayRef> dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnStdMeanCorrection, NPUNativeFunctions::std_mean(self, dim, correction, keepdim));
  c10::SmallVector<int64_t, SIZE> real_dim;
  if (dim.has_value()) {
    real_dim = array_to_small_vector(dim.value());
  }
  auto output_size = reduce_ops_npu_output_size(self, real_dim, keepdim);

  at::Tensor std_out = OpPreparation::apply_tensor_without_format(self, output_size);
  at::Tensor mean_out = OpPreparation::apply_tensor_without_format(self, output_size);

  auto real_correction = correction.value_or(1);
  EXEC_NPU_CMD(aclnnStdMeanCorrection, self, dim, real_correction, keepdim, std_out, mean_out);
  return std::tie(std_out, mean_out);
}

} // namespace native
} // namespace at_npu

