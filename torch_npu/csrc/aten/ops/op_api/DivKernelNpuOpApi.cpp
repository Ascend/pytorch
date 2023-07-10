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

void check_rounding_mode_npu(c10::optional<c10::string_view> rounding_mode) {
  TORCH_CHECK((!rounding_mode.has_value() || *rounding_mode == "trunc" || *rounding_mode == "floor"),
      "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
      "but found '", *rounding_mode, "'");
}

at::Tensor& div_out_npu_opapi_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
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

at::Tensor& NPUNativeOpApiFunctions::div_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnDivs, NPUNativeFunctions::div_out(self, other, result));
  DO_COMPATIBILITY(aclnnDiv, NPUNativeFunctions::div_out(self, other, result));
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
  OpPreparation::CheckOut({self}, result, result_type, output_size);

  // calculate the output result of the NPU
  div_out_npu_opapi_nocheck(self_cp, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::div_out(const at::Tensor& self, const at::Tensor& other,
                                             c10::optional<c10::string_view> rounding_mode, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnDivMods, NPUNativeFunctions::div_out(self, other, rounding_mode, result));
  DO_COMPATIBILITY(aclnnDivMod, NPUNativeFunctions::div_out(self, other, rounding_mode, result));
  if (rounding_mode.has_value() && *rounding_mode != "floor" && *rounding_mode != "trunc") {
    TORCH_CHECK(false,
                "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
                "but found '",
                *rounding_mode, "'");
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, result_type);
  OpPreparation::CheckOut({self}, result, result.scalar_type(), outputSize);

  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = 2;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = 1;
  }
  // calculate the output result of the NPU
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnDivMods, self_cp, others, mode, result);
  } else {
    EXEC_NPU_CMD(aclnnDivMod, self_cp, other, mode, result);
  }
  return result;
}

at::Tensor NPUNativeOpApiFunctions::div(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnDivs, NPUNativeFunctions::div(self, other));
  DO_COMPATIBILITY(aclnnDiv, NPUNativeFunctions::div(self, other));
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor outputTensor = isSelfWrapped ? other : self;
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, high_type);

  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  // construct the output tensor of the NPU
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(outputSize, outputTensor.options().dtype(high_type));

  // calculate the output result of the NPU
  div_out_npu_opapi_nocheck(self_cp, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::div(const at::Tensor& self, const at::Tensor& other,
                                        c10::optional<c10::string_view> rounding_mode) {
  DO_COMPATIBILITY(aclnnDivMods, NPUNativeFunctions::div(self, other, rounding_mode));
  DO_COMPATIBILITY(aclnnDivMod, NPUNativeFunctions::div(self, other, rounding_mode));
  if (rounding_mode.has_value() && *rounding_mode != "floor" && *rounding_mode != "trunc") {
    TORCH_CHECK(false,
                "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
                "but found '",
                *rounding_mode, "'");
  }

  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor outputTensor = isSelfWrapped ? other : self;

  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, high_type);

  // construct the output tensor of the NPU
  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = 2;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = 1;
  } else {
    if (isIntegralType(high_type, true)) {
      high_type = at::ScalarType::Float;
    }
  }
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(outputSize, outputTensor.options().dtype(high_type));

  // executing the NPU operator
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnDivMods, self_cp, others, mode, result);
  } else {
    EXEC_NPU_CMD(aclnnDivMod, self_cp, other, mode, result);
  }
  return result;
}

static at::Tensor& inplace_div_out_npu_no_check(at::Tensor& self, const at::Tensor& other) {
  // check if other scalar tensor
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnInplaceDivs, self, others);
  } else {
    EXEC_NPU_CMD(aclnnInplaceDiv, self, other);
  }
  return self;
}

static at::Tensor& inplace_div_out_mode_npu_no_check(at::Tensor& self, const at::Tensor& other, int mode) {
  // check if other scalar tensor
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar others = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(other);
    EXEC_NPU_CMD(aclnnInplaceDivMods, self, others, mode);
  } else {
    EXEC_NPU_CMD(aclnnInplaceDivMod, self, other, mode);
  }
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::div_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceDivs, NPUNativeFunctions::div_(self, other));
  DO_COMPATIBILITY(aclnnInplaceDiv, NPUNativeFunctions::div_(self, other));

  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  inplace_div_out_npu_no_check(self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::div_(at::Tensor& self, const at::Tensor& other,
                                          c10::optional<c10::string_view> rounding_mode) {
  DO_COMPATIBILITY(aclnnInplaceDivMods, NPUNativeFunctions::div_(self, other, rounding_mode));
  DO_COMPATIBILITY(aclnnInplaceDivMod, NPUNativeFunctions::div_(self, other, rounding_mode));
  check_rounding_mode_npu(rounding_mode);
  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = 2;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = 1;
  }
  inplace_div_out_mode_npu_no_check(self, other, mode);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::div(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnDivs, NPUNativeFunctions::div(self, other));
  auto outputSize = input_same_output_size(self);
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(high_type));
  EXEC_NPU_CMD(aclnnDivs, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::div(const at::Tensor& self, const at::Scalar& other,
                                        c10::optional<c10::string_view> rounding_mode) {
  DO_COMPATIBILITY(aclnnDivMods, NPUNativeFunctions::div(self, other, rounding_mode));
  check_rounding_mode_npu(rounding_mode);
  auto outputSize = input_same_output_size(self);
  at::ScalarType high_type = at::native::result_type(self, other);
  // construct the output tensor of the NPU
  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = 2;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = 1;
  } else {
    if (isIntegralType(high_type, true)) {
      high_type = at::ScalarType::Float;
    }
  }
  at::Tensor result = 
      OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(high_type));
  EXEC_NPU_CMD(aclnnDivMods, self, other, mode, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::div_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceDivs, NPUNativeFunctions::div_(self, other));
  EXEC_NPU_CMD(aclnnInplaceDivs, self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::div_(at::Tensor& self, const at::Scalar& other,
                                          c10::optional<c10::string_view> rounding_mode) {
  DO_COMPATIBILITY(aclnnInplaceDivMods, NPUNativeFunctions::div_(self, other, rounding_mode));
  check_rounding_mode_npu(rounding_mode);
  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = 2;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = 1;
  }
  EXEC_NPU_CMD(aclnnInplaceDivMods, self, other, mode);
  return self;
}

}  // namespace native
}  // namespace at_npu
