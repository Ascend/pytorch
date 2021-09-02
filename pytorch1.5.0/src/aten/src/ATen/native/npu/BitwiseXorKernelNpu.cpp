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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bitwise_xor_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Scalar other) {
  // executing the NPU operator
  Tensor selfInput = (self.dtype() == at::ScalarType::Bool) ? self.to(at::ScalarType::Int) : self;
  result = (result.dtype() == at::ScalarType::Bool) ? result.to(at::ScalarType::Int) : result;

  OpCommand cmd;
  cmd.Name("BitwiseXor")
      .Input(selfInput)
      .Input(other, selfInput.scalar_type())
      .Output(result)
      .Run();

  return (result = (self.dtype() == at::ScalarType::Bool) ? result.to(at::ScalarType::Bool) : result);
}

Tensor& bitwise_xor_out_npu(
    Tensor& result,
    const Tensor& self,
    const Scalar other) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

Tensor& bitwise_xor_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);

  Tensor selfInput  = (self.dtype() == at::ScalarType::Bool) ? self.to(at::ScalarType::Int) : self;
  Tensor otherInput = (other.dtype() == at::ScalarType::Bool) ? other.to(at::ScalarType::Int) : other;
  result = (result.dtype() == at::ScalarType::Bool) ? result.to(at::ScalarType::Int) : result;

  if (otherInput.dim() == 0 && !otherInput.is_npu()) {
    bitwise_xor_out_npu(result, selfInput, otherInput.item());
  } else if (selfInput.dim() == 0 && !selfInput.is_npu()) {
    bitwise_xor_out_npu(result, otherInput, selfInput.item());
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

  return (result = (self.dtype() == at::ScalarType::Bool) ? result.to(at::ScalarType::Bool) : result);
}

Tensor& bitwise_xor_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  Tensor outputTensor;
  if (not isSelfWrapped) {
    outputTensor = self;
  } else {
    outputTensor = other;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(outputTensor),
      outputTensor.scalar_type(),
      outputSize);

  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

Tensor bitwise_xor_npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  Tensor outputTensor;
  if (not isSelfWrapped) {
    outputTensor = self;
  } else {
    outputTensor = other;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);
  // calculate the output result of the NPU
  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

Tensor bitwise_xor_npu(const Tensor& self, Scalar other) {
  Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  bitwise_xor_out_npu_nocheck(result, self, other);

  return result;
}

Tensor& bitwise_xor_npu_(Tensor& self, const Tensor& other) {
  OpPreparation::CheckMemory({self, other}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = bitwise_xor_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_xor_out_npu_nocheck(self, self, other);
  }

  return self;
}

Tensor& bitwise_xor_npu_(Tensor& self, Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = bitwise_xor_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_xor_out_npu_nocheck(self, self, other);
  }

  return self;
}

} // namespace native
} // namespace at