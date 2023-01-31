// Copyright (c) 2022 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& remainder_out_scalar_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other) {
  OpCommand cmd;
  cmd.Name("FloorMod")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::remainder_out(
    const at::Tensor& self,
    const at::Scalar other,
    at::Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);
  remainder_out_scalar_npu_nocheck(result, self, other);
  return result;
}

at::Tensor& remainder_out_tensor_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  if (other.dim() == 0) {
    NPUNativeFunctions::remainder_out(self, other.item(), result);
  } else {
    OpCommand cmd;
    cmd.Name("FloorMod")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

at::Tensor& NPUNativeFunctions::remainder_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  at::Tensor outputTensor = CalcuOpUtil::IsScalarWrappedToTensor(self) ? other : self;
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(outputTensor),
      self.scalar_type(),
      outputSize);
  remainder_out_tensor_npu_nocheck(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::remainder(const at::Tensor& self, const at::Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor outputTensor = isSelfWrapped ? other : self;

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::GetTensorNpuFormat(outputTensor));

  // calculate the output result of the NPU
  remainder_out_tensor_npu_nocheck(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::remainder(const at::Tensor& self, at::Scalar other) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  remainder_out_scalar_npu_nocheck(result, self, other);

  return result;
}

at::Tensor& NPUNativeFunctions::remainder_(at::Tensor& self, const at::Tensor& other) {
  at::SmallVector<at::Tensor, N> inputs = {other};
  at::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = remainder_out_tensor_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    remainder_out_tensor_npu_nocheck(self, self, other);
  }

  return self;
}


at::Tensor& NPUNativeFunctions::remainder_(at::Tensor& self, at::Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = remainder_out_scalar_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    remainder_out_scalar_npu_nocheck(self, self, other);
  }
  return self;
}

} // namespace native
} // namespace at_npu