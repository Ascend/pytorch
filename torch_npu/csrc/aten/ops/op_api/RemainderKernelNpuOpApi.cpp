// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

static const int64_t SIZE_INT64 = 8;

// get the shape result after broadcast
static at::Tensor remainder_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  return isSelfWrapped ? other : self;
}

// tensor + scalar
at::Tensor& NPUNativeOpApiFunctions::remainder_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result)
{
  DO_COMPATIBILITY(aclnnRemainderTensorScalar, NPUNativeFunctions::remainder_out(self, other, result));
  OpPreparation::CheckOut({self}, result, result.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnRemainderTensorScalar, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::remainder(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnRemainderTensorScalar, NPUNativeFunctions::remainder(self, other));
  at::ScalarType result_type = at::native::result_type(self, other); // promote_type
  at::Tensor result = OpPreparation::ApplyTensor(self, self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnRemainderTensorScalar, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::remainder_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceRemainderTensorScalar, NPUNativeFunctions::remainder_(self, other));
  EXEC_NPU_CMD(aclnnInplaceRemainderTensorScalar, self, other);
  return self;
}


// scalar + tensor
at::Tensor NPUNativeOpApiFunctions::remainder(const at::Scalar& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnRemainderScalarTensor, NPUNativeFunctions::remainder(self, other));
  at::ScalarType result_type = at::native::result_type(self, other); // promote_type
  at::Tensor result = OpPreparation::ApplyTensor(other, other.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnRemainderScalarTensor, self, other, result);
  return result;
}


// tensor + tensor
at::Tensor& NPUNativeOpApiFunctions::remainder_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
  DO_COMPATIBILITY(aclnnRemainderTensorTensor, NPUNativeFunctions::remainder_out(self, other, result));
  auto broadcast_shape = broadcast_ops_npu_output_size(self, other);
  auto self_shape = array_to_small_vector(self.sizes());
  if (broadcast_shape == self_shape) {
    OpPreparation::CheckOut({self, other}, result, result.scalar_type(), self.sizes());
  } else {
    OpPreparation::CheckOut({self, other}, result, result.scalar_type(), other.sizes());
  }

  EXEC_NPU_CMD(aclnnRemainderTensorTensor, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::remainder(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnRemainderTensorTensor, NPUNativeFunctions::remainder(self, other));
  at::Tensor output_tensor = remainder_dest_output(self, other);
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other); // promote_type
  at::Tensor result = OpPreparation::ApplyTensor(output_size, output_tensor.options().dtype(result_type),
      output_tensor);
  EXEC_NPU_CMD(aclnnRemainderTensorTensor, self, other, result);

  return result;
}

at::Tensor& NPUNativeOpApiFunctions::remainder_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceRemainderTensorTensor, NPUNativeFunctions::remainder_(self, other));
  at::ScalarType promote_type = at::native::result_type(self, other);
  EXEC_NPU_CMD(aclnnInplaceRemainderTensorTensor, self, other);

  return self;
}


} // namespace native
} // namespace at_npu
