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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::ge_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGeScalar, NPUNativeFunctions::ge_out(self, other, result));
  auto outputSize = self.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  EXEC_NPU_CMD(aclnnGeScalar, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::ge(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnGeScalar, NPUNativeFunctions::ge(self, other));
  auto options = at::TensorOptions().device(self.device());
  at::Tensor result = NPUNativeFunctions::empty(self.sizes(), at::kBool, options.layout_opt(), options.device_opt());
  EXEC_NPU_CMD(aclnnGeScalar, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::ge_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGeTensor, NPUNativeFunctions::ge_out(self, other, result));
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  EXEC_NPU_CMD(aclnnGeTensor, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::ge(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnGeTensor, NPUNativeFunctions::ge(self, other));
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    DO_COMPATIBILITY(aclnnGeScalar, NPUNativeFunctions::ge(self, other));
    auto options = at::TensorOptions().device(self.device());
    at::Tensor result = NPUNativeFunctions::empty(self.sizes(), at::kBool, options.layout_opt(), options.device_opt());
    const at::Scalar tmpItem = other.item();
    EXEC_NPU_CMD(aclnnGeScalar, self, tmpItem, result);
    return result;
  } else if (self.dim() == 0 && !at_npu::key::isDeviceTensor(self)) {
    DO_COMPATIBILITY(aclnnLessScalar, NPUNativeFunctions::ge(self, other));
    auto options = at::TensorOptions().device(other.device());
    at::Tensor result = NPUNativeFunctions::empty(other.sizes(), at::kBool, options.layout_opt(), options.device_opt());
    const at::Scalar tmpItem = self.item();
    EXEC_NPU_CMD(aclnnLessScalar, other, tmpItem, result);
    return result;
  } else {
    auto options = at::TensorOptions().device(self.device());
    auto outputSize = broadcast_ops_npu_output_size(self, other);
    at::Tensor result = NPUNativeFunctions::empty(outputSize, at::kBool, options.layout_opt(), options.device_opt());
    EXEC_NPU_CMD(aclnnGeTensor, self, other, result);
    return result;
  }
}

at::Tensor& NPUNativeOpApiFunctions::ge_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceGeScalar, NPUNativeFunctions::ge_(self, other));
  EXEC_NPU_CMD(aclnnInplaceGeScalar, self, other);
  return self;
}

at::Tensor &NPUNativeOpApiFunctions::ge_(at::Tensor &self, const at::Tensor &other) {
  DO_COMPATIBILITY(aclnnInplaceGeTensor, NPUNativeFunctions::ge_(self, other));
  if (OpPreparation::IsCPUScalar(other)) {
    return NPUNativeOpApiFunctions::ge_(self, other.item());
  } else {
    OpPreparation::CheckMemory({self, other}, {self});
    EXEC_NPU_CMD(aclnnInplaceGeTensor, self, other);
    return self;
  }
}
} // namespace native
} // namespace at_npu
