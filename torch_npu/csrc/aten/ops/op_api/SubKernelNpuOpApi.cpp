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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

inline void alpha_check_npu(const at::ScalarType dtype, at::Scalar alpha) {
  TORCH_CHECK(isFloatingType(dtype) || alpha.isIntegral(true),
              "For integral input tensors, argument alpha must not be a floating point number.");
}

static at::Tensor &sub_out_npu_nocheck(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar alpha,
    at::Tensor &result) {
  if (OpPreparation::IsCPUScalar(other)) {
    c10::Scalar other_scalar = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnSubs, self, other_scalar, alpha, result);
  } else {
    EXEC_NPU_CMD(aclnnSub, self, other, alpha, result);
  }
  return result;
}

static at::Tensor& inplace_sub_out_npu_no_check(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  if (OpPreparation::IsCPUScalar(other)) {
    c10::Scalar other_scalar = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnInplaceSubs, self, other_scalar, alpha);
  } else {
    EXEC_NPU_CMD(aclnnInplaceSub, self, other, alpha);
  }
  return self;
}

static at::Tensor self_tensor_to_device(const at::Tensor &tensor, const at::ScalarType result_type) {
  if (at_npu::native::CalcuOpUtil::IsScalarWrappedToTensor(tensor)) {
    at::Scalar scalar = CalcuOpUtil::ConvertTensorToScalar(tensor);
    return CalcuOpUtil::CopyScalarToDevice(scalar, result_type);
  }
  return tensor;
}

static at::Tensor sub_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  return is_self_wrapped ? other : self;
}

at::Tensor &NPUNativeOpApiFunctions::sub_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &result) {
  DO_COMPATIBILITY(aclnnSub, NPUNativeFunctions::sub_out(self, other, alpha, result));
  DO_COMPATIBILITY(aclnnSubs, NPUNativeFunctions::sub_out(self, other, alpha, result));
  alpha_check_npu(self.scalar_type(), alpha);
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_converted = self_tensor_to_device(self, result_type);

  OpPreparation::CheckOut(
      {self},
      result,
      result,
      output_size);

  CalcuOpUtil::CheckMemoryOverLaps({self, other}, {result});
  sub_out_npu_nocheck(self_converted, other, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sub(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnSub, NPUNativeFunctions::sub(self, other, alpha));
  DO_COMPATIBILITY(aclnnSubs, NPUNativeFunctions::sub(self, other, alpha));
  alpha_check_npu(self.scalar_type(), alpha);
  at::Tensor output_tensor = sub_dest_output(self, other);
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_converted = self_tensor_to_device(self, result_type);
  auto result = OpPreparation::ApplyTensorWithoutFormat(output_size, output_tensor.options().dtype(result_type));
  sub_out_npu_nocheck(self_converted, other, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sub(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnSubs, NPUNativeFunctions::sub(self, other, alpha));
  alpha_check_npu(self.scalar_type(), alpha);
  auto output_size = input_same_output_size(self);
  at::ScalarType result_type = at::native::result_type(self, other);
  auto result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnSubs, self, other, alpha, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::sub_(
    at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnInplaceSub, NPUNativeFunctions::sub_(self, other, alpha));
  DO_COMPATIBILITY(aclnnInplaceSubs, NPUNativeFunctions::sub_(self, other, alpha));
  alpha_check_npu(self.scalar_type(), alpha);
  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  inplace_sub_out_npu_no_check(self, other, alpha);
  return self;
}

at::Tensor &NPUNativeOpApiFunctions::sub_(
    at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnInplaceSubs, NPUNativeFunctions::sub_(self, other, alpha));
  alpha_check_npu(self.scalar_type(), alpha);
  EXEC_NPU_CMD(aclnnInplaceSubs, self, other, alpha);
  return self;
}
  
} // namespace native
} // namespace at_npu
