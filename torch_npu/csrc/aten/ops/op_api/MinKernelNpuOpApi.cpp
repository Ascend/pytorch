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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::minimum(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnMinimum, NPUNativeFunctions::minimum(self, other));
  auto result_type = at::result_type(self, other);
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(output_size, self.options().dtype(result_type), self);
  EXEC_NPU_CMD(aclnnMinimum, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::minimum_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMinimum, NPUNativeFunctions::minimum_out(self, other, result));
  auto output_size = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self, other}, result, result.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnMinimum, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::min(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnMin, NPUNativeFunctions::min(self));
  at::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  auto output_size = reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  EXEC_NPU_CMD(aclnnMin, self, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
