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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::erf_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnErf, NPUNativeFunctions::erf_out(self, result));
  auto result_size = input_same_output_size(self);
  result.resize_(result_size);
  EXEC_NPU_CMD(aclnnErf, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::erf_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceErf, NPUNativeFunctions::erf_(self));
  EXEC_NPU_CMD(aclnnInplaceErf, self);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::erf(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnErf, NPUNativeFunctions::erf(self));
  at::Tensor result;
  if (self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Long ||
      self.scalar_type() == at::ScalarType::Int) {
    result = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), self.options().dtype(at::kFloat));
  } else {
    result = OpPreparation::ApplyTensorWithoutFormat(self);
  }
  EXEC_NPU_CMD(aclnnErf, self, result);
  return result;
}

} // namespace native
} // namespace at_npu

