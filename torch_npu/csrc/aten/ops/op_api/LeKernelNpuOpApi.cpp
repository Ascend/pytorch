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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::le_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLeTensor, NPUNativeFunctions::le_out(self, other, result));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self}, result, at::kBool, outputSize);
  EXEC_NPU_CMD(aclnnLeTensor, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::le_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLeScalar, NPUNativeFunctions::le_out(self, other, result));
  auto outputSize = self.sizes();
  OpPreparation::CheckOut({self}, result, at::kBool, outputSize);

  EXEC_NPU_CMD(aclnnLeScalar, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::le(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnLeTensor, NPUNativeFunctions::le(self, other));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnLeTensor, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::le(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnLeScalar, NPUNativeFunctions::le(self, other));
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnLeScalar, self, other, result);  
  return result;
}

}  // namespace native
}  // namespace at_npu
