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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu{
namespace native{

at::Tensor &NPUNativeOpApiFunctions::sum_out(
    const at::Tensor &self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor &result) {
  DO_COMPATIBILITY(aclnnReduceSum, NPUNativeFunctions::sum_out(self, dim, keepdim, dtype, result));
  auto output_size = sum_npu_output_size(self, dim, keepdim);
  auto res_type = dtype.has_value() ? dtype.value() : result.scalar_type();
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  auto des_dim = ConvertType(dim);
  EXEC_NPU_CMD(aclnnReduceSum, self, des_dim, keepdim, res_type, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::sum_out(
    const at::Tensor &self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor &result) {
  DO_COMPATIBILITY(aclnnReduceSum, NPUNativeFunctions::sum_out(self, dim, keepdim, dtype, result));
  return NPUNativeOpApiFunctions::sum_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

at::Tensor NPUNativeOpApiFunctions::sum(
    const at::Tensor &self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  DO_COMPATIBILITY(aclnnReduceSum, NPUNativeFunctions::sum(self, dim, keepdim, dtype));
  auto output_size = reduce_ops_npu_output_size(self, dim, keepdim);
  auto self_size = self.sizes();
  auto out_type = self.scalar_type();

  if (dtype.has_value()) {
    out_type = dtype.value();
  } else if (isIntegralType(out_type, true)) {
    out_type = at::kLong;
  }

  for (int64_t i = 0; i < self_size.size(); i++) {
    if (self_size[i] == 0) {
      return at::zeros(output_size, self.options().dtype(out_type));
    }
  }

  at::Tensor result = OpPreparation::ApplyTensor(output_size, self.options().dtype(out_type), self);
  auto des_dim = ConvertType(dim);
  EXEC_NPU_CMD(aclnnReduceSum, self, des_dim, keepdim, out_type, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sum(
    const at::Tensor &self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  DO_COMPATIBILITY(aclnnReduceSum, NPUNativeFunctions::sum(self, dim, keepdim, dtype));
  return NPUNativeOpApiFunctions::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor NPUNativeOpApiFunctions::sum(const at::Tensor &self, c10::optional<c10::ScalarType> dtype) {
  DO_COMPATIBILITY(aclnnReduceSum, NPUNativeFunctions::sum(self, dtype));
  return NPUNativeOpApiFunctions::sum(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}

} // namespace native
} // namespace at_npu
