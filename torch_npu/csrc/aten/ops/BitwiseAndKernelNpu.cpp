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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& bitwise_and_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other) {
  // executing the NPU operator
  string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalAnd" : "BitwiseAnd";

  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor & NPUNativeFunctions::bitwise_and_out(const at::Tensor & self, at::Scalar other, at::Tensor & result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(),
      self.sizes());

  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor& bitwise_and_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    NPUNativeFunctions::bitwise_and_out(self, other.item(), result);
  } else if (self.dim() == 0 && !at_npu::key::isDeviceTensor(self)) {
    NPUNativeFunctions::bitwise_and_out(other, self.item(), result);
  } else {
    // executing the NPU operator
    string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalAnd" : "BitwiseAnd";

    OpCommand cmd;
    cmd.Name(real_op_name)
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

at::Tensor & NPUNativeFunctions::bitwise_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & result) {
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

  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::bitwise_and(const at::Tensor& self, const at::Tensor& other) {
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
  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::bitwise_and(const at::Tensor& self, at::Scalar other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output at::Tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

  // calculate the output result of the NPU
  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor& NPUNativeFunctions::bitwise_and_(at::Tensor& self, const at::Tensor& other) {
  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = bitwise_and_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_and_out_npu_nocheck(self, self, other);
  }

  return self;
}

at::Tensor& NPUNativeFunctions::bitwise_and_(at::Tensor& self, at::Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = bitwise_and_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_and_out_npu_nocheck(self, self, other);
  }

  return self;
}
} // namespace native
} // namespace at_npu 