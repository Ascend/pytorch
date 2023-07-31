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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::fmod_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnFmodTensor, NPUNativeFunctions::fmod_out(self, other, result));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  result.resize_(outputSize);
  EXEC_NPU_CMD(aclnnFmodTensor, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::fmod(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnFmodTensor, NPUNativeFunctions::fmod(self, other));
  // calculate the output size and output dtype
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType resultType = at::native::result_type(self, other);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(resultType));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnFmodTensor, self, other, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::fmod_out(const at::Tensor &self, const at::Scalar& other, at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnFmodScalar, NPUNativeFunctions::fmod_out(self, other, result));
  auto outputSize = self.sizes();
  result.resize_(outputSize);
  EXEC_NPU_CMD(aclnnFmodScalar, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::fmod(const at::Tensor &self, const at::Scalar& other)
{
  DO_COMPATIBILITY(aclnnFmodScalar, NPUNativeFunctions::fmod(self, other));
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnFmodScalar, self, other, result);  
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::fmod_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceFmodTensor, NPUNativeFunctions::fmod_(self, other));
  EXEC_NPU_CMD(aclnnInplaceFmodTensor, self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::fmod_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceFmodScalar, NPUNativeFunctions::fmod_(self, other));
  EXEC_NPU_CMD(aclnnInplaceFmodScalar, self, other);
  return self;
}

}  // namespace native
}  // namespace at_npu
