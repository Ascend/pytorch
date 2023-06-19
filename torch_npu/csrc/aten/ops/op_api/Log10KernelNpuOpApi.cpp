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

at::Tensor& NPUNativeOpApiFunctions::log10_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLog10, NPUNativeFunctions::log10_out(self, result));
  OpPreparation::CheckOut({self}, result, self.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnLog10, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::log10(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnLog10, NPUNativeFunctions::log10(self));
  // construct the output tensor of the NPU
  at::Tensor result;
  if (self.scalar_type() != at::ScalarType::Float || self.scalar_type() != at::ScalarType::Half) {
    result = OpPreparation::ApplyTensor(self, self.options().dtype(at::kFloat));
  } else {
    result = OpPreparation::ApplyTensor(self);
  }
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLog10, self, result);

  return result;
}

at::Tensor& NPUNativeOpApiFunctions::log10_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceLog10, NPUNativeFunctions::log10_(self));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnInplaceLog10, self);

  return self;
}

} // namespace native
} // namespace at_npu
