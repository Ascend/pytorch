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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor &sub_scalar_out_npu(
    at::Tensor &result,
    const at::Tensor &self,
    at::Scalar other,
    at::Scalar alpha) {
  // other*alpha
  float otherValue = CalcuOpUtil::GetScalarFloatValue(other);
  float alphaValue = CalcuOpUtil::GetScalarFloatValue(alpha);
  at::Scalar scalarValue(otherValue * alphaValue);

  OpCommand cmd;
  cmd.Name("Sub")
      .Input(self)
      .Input(scalarValue, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor &sub_out_npu_nocheck(
    at::Tensor &result,
    const at::Tensor &self,
    const at::Tensor &other,
    at::Scalar alpha) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    sub_scalar_out_npu(result, self, other.item(), alpha);
  } else {
    at::Tensor otherMulResult = other;
    if (!CalcuOpUtil::IsScalarOne(alpha)) {
      otherMulResult = at::mul(other, alpha);
    }

    OpCommand cmd;
    cmd.Name("Sub")
        .Expect(unified_result)
        .Input(self)
        .Input(otherMulResult)
        .Output(result)
        .Run();
  }

  return result;
}

at::Tensor &NPUNativeFunctions::sub_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Scalar alpha,
    at::Tensor &result) {
  at::Tensor output_tensor = CalcuOpUtil::IsScalarWrappedToTensor(self) ? other : self;
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = (self.scalar_type() != result_type && !CalcuOpUtil::IsScalarWrappedToTensor(self)) ?
      NPUNativeFunctions::npu_dtype_cast(self, result_type) : self;
  at::Tensor other_cp = (other.scalar_type() != result_type && !CalcuOpUtil::IsScalarWrappedToTensor(other)) ?
      NPUNativeFunctions::npu_dtype_cast(other, result_type) : other;
  OpPreparation::CheckOut(
      {self_cp},
      result,
      CalcuOpUtil::GetTensorNpuFormat(output_tensor),
      result_type,
      output_size);
  sub_out_npu_nocheck(result, self_cp, other_cp, alpha);

  return result;
}

at::Tensor NPUNativeFunctions::sub(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor output_tensor = isSelfWrapped ? other : self;

  // calculate the output size
  auto output_size = broadcast_ops_npu_output_size(self, other);

  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor self_cp = (self.scalar_type() != result_type && !CalcuOpUtil::IsScalarWrappedToTensor(self)) ?
      NPUNativeFunctions::npu_dtype_cast(self, result_type) : self;
  at::Tensor other_cp = (other.scalar_type() != result_type && !CalcuOpUtil::IsScalarWrappedToTensor(other)) ?
      NPUNativeFunctions::npu_dtype_cast(other, result_type) : other;

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      output_size,
      output_tensor.options().dtype(result_type),
      CalcuOpUtil::GetTensorNpuFormat(output_tensor));

  // calculate the output result of the NPU
  sub_out_npu_nocheck(result, self_cp, other_cp, alpha);

  return result;
}

at::Tensor NPUNativeFunctions::sub(const at::Tensor &self, at::Scalar other, at::Scalar alpha) {
  // calculate the output size
  auto output_size = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      output_size, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

  // calculate the output result of the NPU
  sub_scalar_out_npu(result, self, other, alpha);

  return result;
}

at::Tensor &NPUNativeFunctions::sub_(at::Tensor &self, const at::Tensor &other, at::Scalar alpha) {
  at::ScalarType result_type = at::native::result_type(self, other);
  at::ScalarType self_type = self.scalar_type();
  TORCH_CHECK(canCast(result_type, self_type), "result type ", result_type,
      " can't be cast to the desired output type ", self_type);
  at::Tensor self_cp = (self_type != result_type && !CalcuOpUtil::IsScalarWrappedToTensor(self)) ?
      NPUNativeFunctions::npu_dtype_cast(self, result_type) : self;
  at::Tensor other_cp = (other.scalar_type() != result_type && !CalcuOpUtil::IsScalarWrappedToTensor(other)) ?
      NPUNativeFunctions::npu_dtype_cast(other, result_type) : other;

  c10::SmallVector<at::Tensor, N> inputs = {self_cp, other_cp};
  c10::SmallVector<at::Tensor, N> outputs = {self_cp};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  if (!NpuUtils::check_match(&self_cp)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self_cp);
    at::Tensor result = sub_out_npu_nocheck(contiguous_self, contiguous_self, other_cp, alpha);
    NpuUtils::format_fresh_view(self_cp, result);
  } else {
    sub_out_npu_nocheck(self_cp, self_cp, other_cp, alpha);
  }

  if (self_type == result_type) {
    self = self_cp;
  } else {
    self.copy_(self_cp);
  }
  return self;
}

at::Tensor &NPUNativeFunctions::sub_(at::Tensor &self, at::Scalar other, at::Scalar alpha) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    at::Tensor result = sub_scalar_out_npu(contiguous_self, contiguous_self, other, alpha);
    NpuUtils::format_fresh_view(self, result);
  } else {
    sub_scalar_out_npu(self, self, other, alpha);
  }

  return self;
}

} // namespace native
} // namespace at_npu
