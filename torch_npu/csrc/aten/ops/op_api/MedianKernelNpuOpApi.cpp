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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::median(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnMedian, NPUNativeFunctions::median(self));
  at::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  auto output_size = reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnMedian, self, result);
  return result;
}

std::tuple<at::Tensor,at::Tensor> NPUNativeOpApiFunctions::median(const at::Tensor& self,
                                                                  int64_t dim,
                                                                  bool keepdim) {
  DO_COMPATIBILITY(aclnnMedianDim, NPUNativeFunctions::median(self, dim, keepdim));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor values = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  at::Tensor indices = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kLong));
  EXEC_NPU_CMD(aclnnMedianDim, self, dim, keepdim, values, indices);
  return std::tie(values, indices);
}

std::tuple<at::Tensor,at::Tensor> NPUNativeOpApiFunctions::median(const at::Tensor& self,
                                                                  at::Dimname dim,
                                                                  bool keepdim) {
  DO_COMPATIBILITY(aclnnMedianDim, NPUNativeFunctions::median(self, dim, keepdim));
  int64_t real_dim = dimname_to_position(self, dim);
  at::SmallVector<int64_t, SIZE> dims = {real_dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor values = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  at::Tensor indices = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kLong));
  EXEC_NPU_CMD(aclnnMedianDim, self, real_dim, keepdim, values, indices);
  return std::tie(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::median_out(const at::Tensor& self,
                                                                         int64_t dim,
                                                                         bool keepdim,
                                                                         at::Tensor& values,
                                                                         at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnMedianDim, NPUNativeFunctions::median_out(self, dim, keepdim, values, indices));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut({self}, values, values.scalar_type(), outputSize);
  OpPreparation::CheckOut({self}, indices, indices.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnMedianDim, self, dim, keepdim, values, indices);
  return std::tie(values, indices);
}

std::tuple<at::Tensor&,at::Tensor&> NPUNativeOpApiFunctions::median_out(const at::Tensor& self,
                                                                        at::Dimname dim,
                                                                        bool keepdim,
                                                                        at::Tensor& values,
                                                                        at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnMedianDim, NPUNativeFunctions::median_out(self, dim, keepdim, values, indices));
  int64_t real_dim = dimname_to_position(self, dim);
  at::SmallVector<int64_t, SIZE> dims = {real_dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut({self}, values, values.scalar_type(), outputSize);
  OpPreparation::CheckOut({self}, indices, indices.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnMedianDim, self, real_dim, keepdim, values, indices);
  return std::tie(values, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::nanmedian(const at::Tensor &self, int64_t dim, bool keepdim) {
  DO_COMPATIBILITY(aclnnNanMedianDim, NPUNativeFunctions::nanmedian(self, dim, keepdim));
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  at::Tensor output = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  at::Tensor indices = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kLong));
  EXEC_NPU_CMD(aclnnNanMedianDim, self, dim, keepdim, output, indices);
  return std::tie(output, indices);
}

}  // namespace native
}  // namespace at_npu
