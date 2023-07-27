// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& NPUNativeOpApiFunctions::xlogy_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnXLogYTensor, NPUNativeFunctions::xlogy_out(self, other, result));
  auto output_size = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self, other}, result, result, output_size);
  EXEC_NPU_CMD(aclnnXLogYTensor, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::xlogy_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnXLogYScalarOther, NPUNativeFunctions::xlogy_out(self, other, result));
  OpPreparation::CheckOut({self}, result, result, self.sizes());
  EXEC_NPU_CMD(aclnnXLogYScalarOther, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::xlogy_out(const at::Scalar& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnXLogYScalarSelf, NPUNativeFunctions::xlogy_out(self, other, result));
  OpPreparation::CheckOut({other}, result, result, other.sizes());
  EXEC_NPU_CMD(aclnnXLogYScalarSelf, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::xlogy(const at::Tensor& self, const at::Tensor& other) {
    DO_COMPATIBILITY(aclnnXLogYTensor, NPUNativeFunctions::xlogy(self, other));
    // calculate the output size
    auto output_size = broadcast_ops_npu_output_size(self, other);
    // construct the output tensor of the NPU
    auto result_type = at::result_type(self, other);
    result_type = (isIntegralType(result_type, true)) ?
                  at::kFloat : result_type;
    at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size,
                                                                self.options().dtype(result_type));
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnXLogYTensor, self, other, result);
    return result;
}

at::Tensor NPUNativeOpApiFunctions::xlogy(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnXLogYScalarOther, NPUNativeFunctions::xlogy(self, other));
  at::ScalarType result_type = at::result_type(self, other);
  result_type = (isIntegralType(result_type, true)) ?
                  at::kFloat : result_type;
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnXLogYScalarOther, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::xlogy(const at::Scalar& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnXLogYScalarSelf, NPUNativeFunctions::xlogy(self, other));
  at::ScalarType result_type = at::result_type(self, other);
  result_type = (isIntegralType(result_type, true)) ?
                  at::kFloat : result_type;  
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(other.sizes(), other.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnXLogYScalarSelf, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::xlogy_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceXLogYTensor, NPUNativeFunctions::xlogy_(self, other));
  CalcuOpUtil::CheckMemoryOverLaps({self, other}, {self});
  EXEC_NPU_CMD(aclnnInplaceXLogYTensor, self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::xlogy_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceXLogYScalarOther, NPUNativeFunctions::xlogy_(self, other));
  EXEC_NPU_CMD(aclnnInplaceXLogYScalarOther, self, other);
  return self;
}

} // namespace native
} // namespace at_npu

