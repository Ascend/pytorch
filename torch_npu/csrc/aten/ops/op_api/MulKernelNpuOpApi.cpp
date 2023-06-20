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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

static at::Tensor self_tensor_to_device(const at::Tensor& tensor, const at::ScalarType result_type) {
  if (at_npu::native::CalcuOpUtil::IsScalarWrappedToTensor(tensor)) {
    at::Scalar scalar = tensor.item();
    return CalcuOpUtil::CopyScalarToDevice(scalar, result_type);
  }
  return tensor;
}

at::Tensor& inplace_mul_out_npu_no_check(at::Tensor& self, const at::Tensor& other) {
  // check if other scalar tensor
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar other_scalar = other.item();
    EXEC_NPU_CMD(aclnnInplaceMuls, self, other_scalar);
  } else {
    EXEC_NPU_CMD(aclnnInplaceMul, self, other);
  }
  return self;
}

at::Tensor& mul_out_npu_no_check(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  // check if other scalar tensor
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    c10::Scalar other_scalar = other.item();
    EXEC_NPU_CMD(aclnnMuls, self, other_scalar, result);
  } else {
    EXEC_NPU_CMD(aclnnMul, self, other, result);
  }
  return result;
}

static at::Tensor mul_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  return isSelfWrapped ? other : self;
}

at::Tensor& NPUNativeOpApiFunctions::mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMul, NPUNativeFunctions::mul_out(self, other, result));
  DO_COMPATIBILITY(aclnnMuls, NPUNativeFunctions::mul_out(self, other, result));
  // calculate the output size
  at::Tensor output_tensor = mul_dest_output(self, other);
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, result_type);

  OpPreparation::CheckOut({self}, result, CalcuOpUtil::GetTensorNpuFormat(output_tensor), result.scalar_type(),
                          output_size);

  // calculate the output result of the NPU
  mul_out_npu_no_check(self_cp, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::mul(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnMul, NPUNativeFunctions::mul(self, other));
  DO_COMPATIBILITY(aclnnMuls, NPUNativeFunctions::mul(self, other));
  // calculate the output size
  at::Tensor output_tensor = mul_dest_output(self, other);
  auto output_size = broadcast_ops_npu_output_size(self, other);

  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, result_type);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(output_size, output_tensor.options().dtype(result_type),
                                                           CalcuOpUtil::GetTensorNpuFormat(output_tensor));

  // calculate the output result of the NPU
  mul_out_npu_no_check(self_cp, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::mul(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnMuls, NPUNativeFunctions::mul(self, other));
  auto output_size = input_same_output_size(self);
  at::ScalarType result_type = at::native::result_type(self, other);
  // construct the output tensor of the Npu
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(output_size, self.options().dtype(result_type),
                                                           CalcuOpUtil::GetTensorNpuFormat(self));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnMuls, self, other, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::mul_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceMul, NPUNativeFunctions::mul_(self, other));
  DO_COMPATIBILITY(aclnnInplaceMuls, NPUNativeFunctions::mul_(self, other));
  TORCH_CHECK(at_npu::key::isDeviceTensor(self), "Inplace tensor self must be NPU-Tensor.");

  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  inplace_mul_out_npu_no_check(self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::mul_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceMuls, NPUNativeFunctions::mul_(self, other));
  TORCH_CHECK(at_npu::key::isDeviceTensor(self), "Inplace tensor self must be NPU-Tensor.");
  EXEC_NPU_CMD(aclnnInplaceMuls, self, other);
  return self;
}

}  // namespace native
}  // namespace at_npu
