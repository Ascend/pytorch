// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/NPUBridge.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {


at::Tensor & NPUNativeOpApiFunctions::bitwise_and_out(const at::Tensor & self, const at::Scalar& other, at::Tensor & result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(),
      self.sizes());

  auto other_tensor = CalcuOpUtil::CopyScalarToDevice(other, self.scalar_type());
  EXEC_NPU_CMD(aclnnBitwiseAndTensorOut, self, other_tensor, result);

  return result;
}

at::Tensor & NPUNativeOpApiFunctions::bitwise_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & result) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  at::Tensor ref_tensor;
  if (isSelfWrapped) {
    ref_tensor = other;
  } else {
    ref_tensor = self;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(ref_tensor),
      ref_tensor.scalar_type(),
      outputSize);

  EXEC_NPU_CMD(aclnnBitwiseAndTensorOut, self, other, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::bitwise_and(const at::Tensor& self, const at::Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  at::Tensor ref_tensor;
  if (isSelfWrapped) {
    ref_tensor = other;
  } else {
    ref_tensor = self;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output at::Tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      ref_tensor.options(),
      CalcuOpUtil::GetTensorNpuFormat(ref_tensor));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnBitwiseAndTensorOut, self, other, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::bitwise_and(const at::Tensor& self, const at::Scalar& other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output at::Tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

  auto other_tensor = CalcuOpUtil::CopyScalarToDevice(other, self.scalar_type());
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnBitwiseAndTensorOut, self, other_tensor, result);

  return result;
}

at::Tensor& NPUNativeOpApiFunctions::bitwise_and_(at::Tensor& self, const at::Tensor& other) {
  EXEC_NPU_CMD(aclnnInplaceBitwiseAndTensorOut, self, other);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::bitwise_and_(at::Tensor& self, const at::Scalar& other) {
  auto other_tensor = CalcuOpUtil::CopyScalarToDevice(other, self.scalar_type());
  EXEC_NPU_CMD(aclnnInplaceBitwiseAndTensorOut, self, other_tensor);
  return self;
}
} // namespace native
} // namespace at_npu 