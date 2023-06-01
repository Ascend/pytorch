// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor mul_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  return is_self_wrapped ? other : self;
}

at::Tensor& muls_out_npu(at::Tensor& result, const at::Tensor& self, const at::Scalar other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  if (!other.isFloatingPoint()) {
    unified_result.common_type = self.scalar_type();
    if (self.scalar_type() == at::kBool) {
      unified_result.common_type = other.type();
    }
  }

  OpCommand cmd;
  cmd.Name("Mul")
      .Expect(unified_result)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& mul_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  if (OpPreparation::IsCPUScalar(other)) {
    muls_out_npu(result, self, other.item());
  } else if (OpPreparation::IsCPUScalar(self)) {
    muls_out_npu(result, other, self.item());
  } else {
    auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
    OpCommand cmd;
    cmd.Name("Mul")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}

at::Tensor& NPUNativeFunctions::mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor output_tensor = mul_dest_output(self, other);
  auto high_type = output_tensor.scalar_type();
  auto result_type = result.scalar_type();
  TORCH_CHECK(canCast(high_type, result_type), "result type ", high_type,
      " can't be cast to the desired output type ", result_type);

  auto output_size = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self, other},
      result,
      result,
      output_size);
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  if (self.dtype() == at::kBool && other.dtype() == at::kBool) {
    self_cast = NPUNativeFunctions::npu_dtype_cast(self, at::kFloat);
    other_cast = NPUNativeFunctions::npu_dtype_cast(other, at::kFloat);
  }
  at::Tensor result_cast = (result.scalar_type() != self.scalar_type()) ?
      NPUNativeFunctions::npu_dtype_cast(result, self.scalar_type()) : result;
  mul_out_npu_nocheck(result_cast, self_cast, other_cast);
  if (result.scalar_type() != self.scalar_type()) {
    result_cast = NPUNativeFunctions::npu_dtype_cast(result_cast, result.scalar_type());
    result.copy_(result_cast);
  }
  return result;
}

at::Tensor NPUNativeFunctions::mul(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  if (self.dtype() == c10::ScalarType::Bool && other.dtype() == c10::ScalarType::Bool) {
    self_cast = NPUNativeFunctions::npu_dtype_cast(self, at::kFloat);
    other_cast = NPUNativeFunctions::npu_dtype_cast(other, at::kFloat);
  }

  at::Tensor output_tensor = mul_dest_output(self_cast, other_cast);
  auto output_size = broadcast_ops_npu_output_size(self_cast, other_cast);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      output_size,
      output_tensor.options(),
      CalcuOpUtil::GetTensorNpuFormat(output_tensor));

  mul_out_npu_nocheck(result, self_cast, other_cast);

  if (self.dtype() == c10::ScalarType::Bool && other.dtype() == c10::ScalarType::Bool) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }
  return result;
}

at::Tensor NPUNativeFunctions::mul(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  muls_out_npu(result, self, other);
  return result;
}

at::Tensor& NPUNativeFunctions::mul_(at::Tensor& self, const at::Tensor& other) {
  TORCH_CHECK(at_npu::key::isDeviceTensor(self), "Input1 must be NPU-Tensor");

  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  at::Tensor self_dtype_cast =
      (self.scalar_type() == at::kBool) ? NPUNativeFunctions::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor other_dtype_cast =
      (other.scalar_type() == at::kBool && other.dim() != 0) ? NPUNativeFunctions::npu_dtype_cast(other, at::kFloat) : other;
  if (!NpuUtils::check_match(&self_dtype_cast)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self_dtype_cast);
    at::Tensor result = mul_out_npu_nocheck(contiguous_self, contiguous_self, other_dtype_cast);
    NpuUtils::format_fresh_view(self_dtype_cast, result);
  } else {
    mul_out_npu_nocheck(self_dtype_cast, self_dtype_cast, other_dtype_cast);
  }
  if (self_dtype_cast.scalar_type() != self.scalar_type()) {
    self_dtype_cast = NPUNativeFunctions::npu_dtype_cast(self_dtype_cast, self.scalar_type());
    self.copy_(self_dtype_cast);
  } else {
    self = self_dtype_cast;
  }
  return self;
}

at::Tensor& NPUNativeFunctions::mul_(at::Tensor& self, const at::Scalar& other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    at::Tensor result = muls_out_npu(contiguous_self, contiguous_self, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    muls_out_npu(self, self, other);
  }
  return self;
}
} // namespace native
} // namespace at_npu
