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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& not_out_npu(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("LogicalNot")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& not_out_npu(at::Tensor& result, const at::Scalar self) {
  OpCommand cmd;
  cmd.Name("LogicalNot")
      .Input(self, self.type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& and_out_npu(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  OpCommand cmd;
  cmd.Name("LogicalAnd")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& and_out_npu(at::Tensor& result, const at::Tensor& self, const at::Scalar other) {
  OpCommand cmd;
  cmd.Name("LogicalAnd")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& or_out_npu(at::Tensor& result, const at::Tensor& self, const at::Scalar other) {
  OpCommand cmd;
  cmd.Name("LogicalOr")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& xor_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  // executing the NPU operator
  if (self.dtype() == at::ScalarType::Bool) {
    auto not_self_result = OpPreparation::ApplyTensor(self, outputSize);
    not_out_npu(not_self_result, self);

    auto not_other_result = OpPreparation::ApplyTensor(self, outputSize);
    not_out_npu(not_other_result, other);

    auto not_self_and_other = OpPreparation::ApplyTensor(self, outputSize);
    and_out_npu(not_self_and_other, not_self_result, other);

    auto self_and_not_other = OpPreparation::ApplyTensor(self, outputSize);
    and_out_npu(self_and_not_other, self, not_other_result);

    OpCommand cmd;
    cmd.Name("LogicalOr")
        .Input(not_self_and_other)
        .Input(self_and_not_other)
        .Output(result)
        .Run();
  } else {
    OpCommand cmd;
    cmd.Name("BitwiseXor")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

at::Tensor& xor_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other) {
  // executing the NPU operator
  if (self.dtype() == at::ScalarType::Bool) {
    auto not_self_result = OpPreparation::ApplyTensor(self);
    not_out_npu(not_self_result, self);

    auto not_self_or_other = OpPreparation::ApplyTensor(self);
    or_out_npu(not_self_or_other, not_self_result, other);

   auto not_not_self_or_other = OpPreparation::ApplyTensor(self);
    not_out_npu(not_not_self_or_other, not_self_or_other); 

    auto not_self_and_other = OpPreparation::ApplyTensor(self);
    and_out_npu(not_self_and_other, not_self_result, other);

    OpCommand cmd;
    cmd.Name("LogicalOr")
        .Input(not_self_and_other)
        .Input(not_not_self_or_other)
        .Output(result)
        .Run();

  } else {
    OpCommand cmd;
    cmd.Name("BitwiseXor")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();
  }

  return result;
}

at::Tensor NPUNativeFunctions::__xor__(const at::Tensor& self, const at::Tensor& other) {
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  xor_out_npu(result, self, other);
  return result;
}

at::Tensor NPUNativeFunctions::__xor__(const at::Tensor& self, const at::Scalar other) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  xor_out_npu(result, self, other);

  return result;
}
} // namespace native
} // namespace at_npu