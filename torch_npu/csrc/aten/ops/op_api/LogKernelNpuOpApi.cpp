// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

at::Tensor& NPUNativeOpApiFunctions::log_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLog, NPUNativeFunctions::log_out(self, result));
  if (!result.is_same(self)) {
    at::ScalarType expext_dtype = self.scalar_type();
    if (self.dtype() == at::kLong || self.dtype() == at::kBool) {
      // int need cast to float
      expext_dtype = at::kFloat;
    }
    OpPreparation::CheckOut({self}, result, expext_dtype, self.sizes());
  }

  OpPreparation::CheckMemory({self}, {result});
  EXEC_NPU_CMD(aclnnLog, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::log(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnLog, NPUNativeFunctions::log(self));
  // construct the output tensor of the NPU
  at::ScalarType expext_dtype = self.scalar_type();
  if (self.dtype() == at::kLong || self.dtype() == at::kBool) {
    // int need cast to float
    expext_dtype = at::kFloat;
  }
  at::Tensor result = OpPreparation::ApplyTensor(
      self.sizes(),
      self.options().dtype(expext_dtype),
      self);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLog, self, result);

  return result;
}

at::Tensor& NPUNativeOpApiFunctions::log_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceLog, NPUNativeFunctions::log_(self));
  EXEC_NPU_CMD(aclnnInplaceLog, self);

  return self;
}

}  // namespace native
}  // namespace at_npu
