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

at::Tensor& NPUNativeOpApiFunctions::lt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLtTensor, NPUNativeFunctions::lt_out(self, other, result));
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  OpPreparation::CheckOut({self}, result, at::kBool, outputSize);

  EXEC_NPU_CMD(aclnnLtTensor, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::lt(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnLtTensor, NPUNativeFunctions::lt(self, other));
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kBool));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLtTensor, self, other, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::lt_out(const at::Tensor &self, const at::Scalar& other, at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnLtScalar, NPUNativeFunctions::lt_out(self, other, result));
  auto outputSize = self.sizes();
  OpPreparation::CheckOut({self}, result, at::kBool, outputSize);

  EXEC_NPU_CMD(aclnnLtScalar, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::lt(const at::Tensor &self, const at::Scalar& other)
{
  DO_COMPATIBILITY(aclnnLtScalar, NPUNativeFunctions::lt(self, other));
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kBool));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLtScalar, self, other, result);  
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::lt_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceLtTensor, NPUNativeFunctions::lt_(self, other));
  EXEC_NPU_CMD(aclnnInplaceLtTensor, self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::lt_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceLtScalar, NPUNativeFunctions::lt_(self, other));
  EXEC_NPU_CMD(aclnnInplaceLtScalar, self, other);
  return self;
}

}  // namespace native
}  // namespace at_npu
