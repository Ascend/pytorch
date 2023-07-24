// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include <ATen/Tensor.h>
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

/**
 * Different from Pytorch1.11 for torch.floor_divide() using truncation division, 
 * hostapi are corrected to use floor division. 
 */
static at::Tensor& floor_divide_out_npu_opapi(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  // executing the NPU operator
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnFloorDivides, self, others, result);
  } else {
    EXEC_NPU_CMD(aclnnFloorDivide, self, other, result);
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

static at::Tensor& inplace_floor_divide_out_npu_opapi(at::Tensor& self, const at::Tensor& other) {
  // executing the NPU operator
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnInplaceFloorDivides, self, others);
  } else {
    EXEC_NPU_CMD(aclnnInplaceFloorDivide, self, other);
  }
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::floor_divide_out(const at::Tensor& self, const at::Tensor& other,
                                                      at::Tensor& result) {
  DO_COMPATIBILITY(aclnnFloorDivides, NPUNativeFunctions::floor_divide_out(self, other, result));
  DO_COMPATIBILITY(aclnnFloorDivide, NPUNativeFunctions::floor_divide_out(self, other, result));
  // calculate the output size
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, result_type);
  OpPreparation::CheckOut({self, other}, result, result.scalar_type(), output_size);

  // calculate the output result of the NPU
  floor_divide_out_npu_opapi(self_cp, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::floor_divide(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnFloorDivides, NPUNativeFunctions::floor_divide(self, other));
  DO_COMPATIBILITY(aclnnFloorDivide, NPUNativeFunctions::floor_divide(self, other));
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor outputTensor = isSelfWrapped ? other : self;
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, high_type);

  // construct the output tensor of the NPU
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(outputSize, outputTensor.options().dtype(high_type));

  // calculate the output result of the NPU
  floor_divide_out_npu_opapi(self_cp, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::floor_divide(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnFloorDivides, NPUNativeFunctions::floor_divide(self, other));
  auto outputSize = input_same_output_size(self);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(high_type));
  EXEC_NPU_CMD(aclnnFloorDivides, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::floor_divide_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceFloorDivides, NPUNativeFunctions::floor_divide_(self, other));
  DO_COMPATIBILITY(aclnnInplaceFloorDivide, NPUNativeFunctions::floor_divide_(self, other));

  OpPreparation::CheckMemory({self, other}, {self});
  inplace_floor_divide_out_npu_opapi(self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::floor_divide_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceFloorDivides, NPUNativeFunctions::floor_divide_(self, other));
  EXEC_NPU_CMD(aclnnInplaceFloorDivides, self, other);
  return self;
}

}  // namespace native
}  // namespace at_npu
