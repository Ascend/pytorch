// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::bitwise_xor_out(const at::Tensor& self, const at::Scalar& other,
                                                     at::Tensor& result) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, NPUNativeFunctions::bitwise_xor_out(self, other, result));

  OpPreparation::CheckOut({self}, result, result, self.sizes());

  EXEC_NPU_CMD(aclnnBitwiseXorScalar, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::bitwise_xor(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, NPUNativeFunctions::bitwise_xor(self, other));

  at::Tensor result;
  if ((self.scalar_type() == at::ScalarType::Bool) && (!other.isBoolean())) {
    result = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), self.options().dtype(at::kLong));
  } else {
    result = OpPreparation::ApplyTensorWithoutFormat(self);
  }

  EXEC_NPU_CMD(aclnnBitwiseXorScalar, self, other, result);
  return result;
}

at::Tensor& bitwise_xor_op_api_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  if (OpPreparation::IsCPUScalar(other)) {
    const at::Scalar &other_value = other.item();
    EXEC_NPU_CMD(aclnnBitwiseXorScalar, self, other_value, result);
  } else if (OpPreparation::IsCPUScalar(self)) {
    const at::Scalar &self_value = self.item();
    EXEC_NPU_CMD(aclnnBitwiseXorScalar, other, self_value, result);
  } else {
    EXEC_NPU_CMD(aclnnBitwiseXorTensor, self, other, result);
  }

  return result;
}

at::Tensor& NPUNativeOpApiFunctions::bitwise_xor_out(const at::Tensor& self, const at::Tensor& other,
                                                     at::Tensor& result) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, NPUNativeFunctions::bitwise_xor_out(self, other, result));
  DO_COMPATIBILITY(aclnnBitwiseXorTensor, NPUNativeFunctions::bitwise_xor_out(self, other, result));

  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self}, result, result, outputSize);

  bitwise_xor_op_api_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::bitwise_xor(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, NPUNativeFunctions::bitwise_xor(self, other));
  DO_COMPATIBILITY(aclnnBitwiseXorTensor, NPUNativeFunctions::bitwise_xor(self, other));

  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType resultType = at::native::result_type(self, other);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(resultType));

  bitwise_xor_op_api_out_npu_nocheck(result, self, other);
  return result;
}

} // namespace native
} // namespace at_npu
