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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::neg_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnNeg, NPUNativeFunctions::neg_out(self, result));
  OpPreparation::CheckOut({self}, result, ACL_FORMAT_ND, self.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnNeg, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::neg(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnNeg, NPUNativeFunctions::neg(self));
  // construct the output tensor of the NPU
  at::Tensor result =
      OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

  EXEC_NPU_CMD(aclnnNeg, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::neg_(at::Tensor &self) {
  DO_COMPATIBILITY(aclnnInplaceNeg, NPUNativeFunctions::neg_(self));
  c10::SmallVector<at::Tensor, N> inputs = {self};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  EXEC_NPU_CMD(aclnnInplaceNeg, self);
  return self;
}
}  // namespace native
}  // namespace at_npu
