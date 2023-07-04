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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::gt_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGtScalar, NPUNativeFunctions::gt_out(self, other, result));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes();
  OpPreparation::CheckOut({self}, result, ACL_FORMAT_ND, result.scalar_type(), outputSize);

  EXEC_NPU_CMD(aclnnGtScalar, formatCastOfSelf, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::gt(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnGtScalar, NPUNativeFunctions::gt(self, other));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfSelf);

  // construct the output tensor of the NPU
  at::Tensor result =
      OpPreparation::ApplyTensorWithFormat(outputSize, formatCastOfSelf.options().dtype(at::kBool), ACL_FORMAT_ND);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnGtScalar, formatCastOfSelf, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::gt_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceGtScalar, NPUNativeFunctions::gt_(self, other));
  EXEC_NPU_CMD(aclnnInplaceGtScalar, self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::gt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGtTensor, NPUNativeFunctions::gt_out(self, other, result));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut({self}, result, ACL_FORMAT_ND, result.scalar_type(), outputSize);

  EXEC_NPU_CMD(aclnnGtTensor, formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::gt(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnGtTensor, NPUNativeFunctions::gt(self, other));
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  // construct the output tensor of the NPU
  at::Tensor result =
      OpPreparation::ApplyTensorWithFormat(outputSize, formatCastOfSelf.options().dtype(at::kBool), ACL_FORMAT_ND);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnGtTensor, formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::gt_(at::Tensor &self, const at::Tensor &other) {
  DO_COMPATIBILITY(aclnnInplaceGtTensor, NPUNativeFunctions::gt_(self, other));
  if (OpPreparation::IsCPUScalar(other)) {
    return NPUNativeOpApiFunctions::gt_(self, other.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        (self.device().type() == at_npu::key::NativeDeviceType ? "npu" : "cpu"),
        " and ",
        (other.device().type() == at_npu::key::NativeDeviceType ? "npu! " : "cpu! "));
    OpPreparation::CheckMemory({self, other}, {self});
    EXEC_NPU_CMD(aclnnInplaceGtTensor, self, other);
    return self;
  }
}

}  // namespace native
}  // namespace at_npu
