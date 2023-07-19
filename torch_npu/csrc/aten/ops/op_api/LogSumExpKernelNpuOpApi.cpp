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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& NPUNativeOpApiFunctions::logsumexp_out(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLogSumExp, NPUNativeFunctions::logsumexp_out(self, dims, keepdim, result));
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut({self}, result, result.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnLogSumExp, self, dims, keepdim, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::logsumexp_out(const at::Tensor& self, at::DimnameList dims, bool keepdim, at::Tensor& result) {
  return logsumexp_out(self, dimnames_to_positions(self, dims), keepdim, result);
}

at::Tensor NPUNativeOpApiFunctions::logsumexp(const at::Tensor& self, at::IntArrayRef dims, bool keepdim) {
  DO_COMPATIBILITY(aclnnLogSumExp, NPUNativeFunctions::logsumexp(self, dims, keepdim));
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  at::ScalarType dst_type = self.scalar_type();
  if (isIntegralType(self.scalar_type(), true)) {
    dst_type = at::kFloat;
  }
  at::Tensor result =  OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(dst_type));
  EXEC_NPU_CMD(aclnnLogSumExp, self, dims, keepdim, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::logsumexp(const at::Tensor& self, at::DimnameList dims, bool keepdim) {
  return logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

}  // namespace native
}  // namespace at_npu
