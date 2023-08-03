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

at::Tensor NPUNativeOpApiFunctions::median(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnMedian, NPUNativeFunctions::median(self));
  at::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  auto output_size = reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnMedian, self, result);
  return result;
}

tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::median_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnMedianDim, NPUNativeFunctions::median_out(self, dim, keepdim, output, indices));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut({self}, output, self.scalar_type(), outputSize);
  OpPreparation::CheckOut({self}, indices, at::ScalarType::Long, outputSize);
  EXEC_NPU_CMD(aclnnMedianDim, self, dim, keepdim, output, indices);
  return std::tie(output, indices);
}

at::Tensor NPUNativeOpApiFunctions::nanmedian(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnNanMedian, NPUNativeFunctions::nanmedian(self));
  at::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  auto output_size = reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnNanMedian, self, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
