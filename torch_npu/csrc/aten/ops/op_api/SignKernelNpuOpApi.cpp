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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor &NPUNativeOpApiFunctions::sign_out(const at::Tensor &self, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnSign, NPUNativeFunctions::sign_out(self, result));
  OpPreparation::CheckOut({self}, result, self);
  EXEC_NPU_CMD(aclnnSign, self, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::sgn_out(const at::Tensor &self, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnSign, NPUNativeFunctions::sgn_out(self, result));
  OpPreparation::CheckOut({self}, result, self);
  EXEC_NPU_CMD(aclnnSign, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sgn(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnSign, NPUNativeFunctions::sgn(self));
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options());
  EXEC_NPU_CMD(aclnnSign, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sign(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnSign, NPUNativeFunctions::sign(self));
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options());
  EXEC_NPU_CMD(aclnnSign, self, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::sign_(at::Tensor &self) {
  DO_COMPATIBILITY(aclnnSign, NPUNativeFunctions::sign_(self));
  EXEC_NPU_CMD(aclnnSign, self, self);
  return self;
}

}  // namespace native
}  // namespace at_npu
