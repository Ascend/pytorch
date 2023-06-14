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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::any_out(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAny, NPUNativeFunctions::any_out(self, dim, keepdim, result));
  c10::SmallVector<int64_t, N> dim_list = {dim};

  // check result for return
  auto output_size = reduce_ops_npu_output_size(self, dim_list, keepdim);
  OpPreparation::CheckOut({self}, result, result, output_size);
  // calculate the output result of the NPU
  at::IntArrayRef dims(dim);
  EXEC_NPU_CMD(aclnnAny, self, dims, keepdim, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::any(const at::Tensor& self, int64_t dim, bool keepdim) {
  DO_COMPATIBILITY(aclnnAny, NPUNativeFunctions::any(self, dim, keepdim));
  // calculate the output size
  at::IntArrayRef dims(dim);
  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);
  auto output_dtype = self.scalar_type() == at::ScalarType::Byte ? at::ScalarType::Byte : at::ScalarType::Bool;
  auto options = self.options().dtype(output_dtype);
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(output_size, options, self);
  EXEC_NPU_CMD(aclnnAny, self, dims, keepdim, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::any_out(
    const at::Tensor& self,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAny, NPUNativeFunctions::any_out(self, result));
  at::SmallVector<int64_t, N> dim_list = CalcuOpUtil::GetDimlistForTensor(self);
  bool keep_dim = false;
  // check result for return
  auto output_size = reduce_ops_npu_output_size(self, dim_list, keep_dim);
  OpPreparation::CheckOut(
      {self},
      result,
      result,
      output_size);

  at::IntArrayRef dims(dim_list);
  EXEC_NPU_CMD(aclnnAny, self, dims, keep_dim, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::any(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnAny, NPUNativeFunctions::any(self));
  at::SmallVector<int64_t, N> dim_list = CalcuOpUtil::GetDimlistForTensor(self);
  bool keep_dim = false;
  auto output_size = reduce_ops_npu_output_size(self, dim_list, keep_dim);
  auto output_dtype = self.scalar_type() == at::ScalarType::Byte ? at::ScalarType::Byte : at::ScalarType::Bool;
  auto options = self.options().dtype(output_dtype);
  
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(output_size, options, self);
  return NPUNativeOpApiFunctions::any_out(self, result);
}

} // namespace native
} // namespace at_npu
