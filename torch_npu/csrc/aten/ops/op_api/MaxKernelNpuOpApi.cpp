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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::maximum_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMaximum, NPUNativeFunctions::maximum_out(self, other, result));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self, other}, result, result, outputSize);
  EXEC_NPU_CMD(aclnnMaximum, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::maximum(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnMaximum, NPUNativeFunctions::maximum(self, other));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options().dtype(high_type), self);
  EXEC_NPU_CMD(aclnnMaximum, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::max(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnMax, NPUNativeFunctions::max(self));
  at::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  auto output_size = reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  EXEC_NPU_CMD(aclnnMax, self, result);
  return result;
}

tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::max_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnMaxDim, NPUNativeFunctions::max_out(self, dim, keepdim, output, indices));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut({self}, output, self.scalar_type(), outputSize);
  OpPreparation::CheckOut({self}, indices, at::ScalarType::Long, outputSize);
  EXEC_NPU_CMD(aclnnMaxDim, self, dim, keepdim, output, indices);
  return std::tie(output, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::max(
    const at::Tensor& self, 
    int64_t dim, 
    bool keepdim) {
  DO_COMPATIBILITY(aclnnMaxDim, NPUNativeFunctions::max(self, dim, keepdim));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor outputs = OpPreparation::ApplyTensor(outputSize, self.options(), self);
  at::Tensor indices = OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::ScalarType::Long), self);
  EXEC_NPU_CMD(aclnnMaxDim, self, dim, keepdim, outputs, indices);
  return std::tie(outputs, indices);
}

}  // namespace native
}  // namespace at_npu
