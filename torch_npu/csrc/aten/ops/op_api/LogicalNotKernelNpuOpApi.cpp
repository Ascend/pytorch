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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::logical_not_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLogicalNot, NPUNativeFunctions::logical_not_out(self, result));
  auto resultDtype = result.scalar_type();
  OpPreparation::CheckOut(
      {self},
      result,
      resultDtype,
      self.sizes());
  EXEC_NPU_CMD(aclnnLogicalNot, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::logical_not(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnLogicalNot, NPUNativeFunctions::logical_not(self));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(
      self.sizes(),
      self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnLogicalNot, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::logical_not_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceLogicalNot, NPUNativeFunctions::logical_not_(self));
  EXEC_NPU_CMD(aclnnInplaceLogicalNot, self);
  return self;
}
} // namespace native
} // namespace at_npu
