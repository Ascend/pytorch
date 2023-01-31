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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& bitwise_xor_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other) {
  // executing the NPU operator
  at::Tensor selfInput = (self.dtype() == at::ScalarType::Bool) ? NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Int) : self;
  result = (result.dtype() == at::ScalarType::Bool) ? NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Int) : result;

  OpCommand cmd;
  cmd.Name("BitwiseXor")
      .Input(selfInput)
      .Input(other, selfInput.scalar_type())
      .Output(result)
      .Run();

  return (result = (self.dtype() == at::ScalarType::Bool) ? NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Bool) : result);
}

at::Tensor& NPUNativeFunctions::bitwise_xor_out(
    const at::Tensor& self,
    const at::Scalar other,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor& bitwise_xor_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);

  at::Tensor selfInput  = (self.dtype() == at::ScalarType::Bool) ? NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Int) : self;
  at::Tensor otherInput = (other.dtype() == at::ScalarType::Bool) ? NPUNativeFunctions::npu_dtype_cast(other, at::ScalarType::Int) : other;
  result = (result.dtype() == at::ScalarType::Bool) ? NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Int) : result;

  if (otherInput.dim() == 0 && !at_npu::key::isDeviceTensor(otherInput)) {
    NPUNativeFunctions::bitwise_xor_out(selfInput, otherInput.item(), result);
  } else if (selfInput.dim() == 0 && !at_npu::key::isDeviceTensor(selfInput)) {
    NPUNativeFunctions::bitwise_xor_out(otherInput, selfInput.item(), result);
  } else {
    // executing the NPU operator
    OpCommand cmd;
    cmd.Name("BitwiseXor")
        .Expect(unified_result)
        .Input(selfInput)
        .Input(otherInput)
        .Output(result)
        .Run();
  }

  return (result = (self.dtype() == at::ScalarType::Bool) ? NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Bool) : result);
}

at::Tensor& NPUNativeFunctions::bitwise_xor_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  at::Tensor outputTensor;
  if (not isSelfWrapped) {
    outputTensor = self;
  } else {
    outputTensor = other;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(outputTensor),
      outputTensor.scalar_type(),
      outputSize);

  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::bitwise_xor(const at::Tensor& self, const at::Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  at::Tensor outputTensor;
  if (not isSelfWrapped) {
    outputTensor = self;
  } else {
    outputTensor = other;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);
  // calculate the output result of the NPU
  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::bitwise_xor(const at::Tensor& self, at::Scalar other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor& NPUNativeFunctions::bitwise_xor_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CheckMemory({self, other}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = bitwise_xor_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_xor_out_npu_nocheck(self, self, other);
  }

  return self;
}

at::Tensor& NPUNativeFunctions::bitwise_xor_(at::Tensor& self, at::Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = bitwise_xor_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_xor_out_npu_nocheck(self, self, other);
  }

  return self;
}

} // namespace native
} // namespace at_npu