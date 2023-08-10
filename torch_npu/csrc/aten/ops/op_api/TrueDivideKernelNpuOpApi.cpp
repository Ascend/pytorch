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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

static at::Tensor& div_out_npu_opapi_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  // executing the NPU operator
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnDivs, self, others, result);
  } else {
    EXEC_NPU_CMD(aclnnDiv, self, other, result);
  }
  return result;
}

static at::Tensor self_tensor_to_device(const at::Tensor& tensor, const at::ScalarType result_type) {
  if (at_npu::native::CalcuOpUtil::IsScalarWrappedToTensor(tensor)) {
    at::Scalar scalar = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(tensor);
    return CalcuOpUtil::CopyScalarToDevice(scalar, result_type);
  }
  return tensor;
}

at::Tensor NPUNativeOpApiFunctions::true_divide(const at::Tensor &self, const at::Tensor &other) {
  DO_COMPATIBILITY(aclnnDivs, NPUNativeFunctions::true_divide(self, other));
  DO_COMPATIBILITY(aclnnDiv, NPUNativeFunctions::true_divide(self, other));
  // calculate the output size
  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor output_tensor = is_self_wrapped ? other : self;
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, high_type);

  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  // construct the output tensor of the NPU
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(output_size, output_tensor.options().dtype(high_type));

  // calculate the output result of the NPU
  div_out_npu_opapi_nocheck(self_cp, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::true_divide(const at::Tensor &self, const at::Scalar &other) {
  DO_COMPATIBILITY(aclnnDivs, NPUNativeFunctions::true_divide(self, other));
  auto output_size = input_same_output_size(self);
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(high_type));
  EXEC_NPU_CMD(aclnnDivs, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::true_divide_out(const at::Tensor& self, const at::Tensor& other,
                                                     at::Tensor& result) {
  DO_COMPATIBILITY(aclnnDivs, NPUNativeFunctions::true_divide_out(self, other, result));
  DO_COMPATIBILITY(aclnnDiv, NPUNativeFunctions::true_divide_out(self, other, result));
  // calculate the output size
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  if (isIntegralType(result_type, true)) {
    result_type = at::ScalarType::Float;
  }
  if (isFloatingType(result.scalar_type())) {
    result_type = result.scalar_type();
  }
  at::Tensor self_cp = self_tensor_to_device(self, result_type);
  OpPreparation::CheckOut({self, other}, result, result_type, output_size);

  // calculate the output result of the NPU
  div_out_npu_opapi_nocheck(self_cp, other, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
