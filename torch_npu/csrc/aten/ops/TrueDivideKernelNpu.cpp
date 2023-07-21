// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
namespace {
at::Tensor& true_div_scalar_out_nocheck(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("Div")
      .Expect(unified_result)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& true_div_scalar_out_nocheck(const at::Scalar& self, const at::Tensor& other, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Div")
      .Input(self, other.scalar_type())
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& true_div_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  if (OpPreparation::IsCPUScalar(other)) {
    true_div_scalar_out_nocheck(self, other.item(), result);
  } else if (OpPreparation::IsCPUScalar(self)) {
    true_div_scalar_out_nocheck(self.item(), other, result);
  } else {
    auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
    OpCommand cmd;
    cmd.Name("Div")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}

at::ScalarType get_divide_high_type(const at::Tensor& self, const at::Tensor& other) {
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::kFloat;
  }
  return high_type;
}
} // namespace

at::Tensor& NPUNativeFunctions::true_divide_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {  
  auto high_type = get_divide_high_type(self, other);
  at::ScalarType result_type = result.scalar_type();
  TORCH_CHECK(canCast(high_type, result_type),
      "result type ", high_type, " can't be cast to the desired output type ", result_type);

  at::Tensor self_temp = (self.scalar_type() == high_type) ? self : self.to(high_type);
  at::Tensor other_temp = (other.scalar_type() == high_type) ? other : other.to(high_type);

  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self_temp) || OpPreparation::IsCPUScalar(self_temp);
  at::Tensor output_tensor = is_self_wrapped ? other_temp : self_temp;
  auto output_size = broadcast_ops_npu_output_size(self_temp, other_temp);
  OpPreparation::CheckOut(
      {self_temp, other_temp},
      result,
      CalcuOpUtil::GetTensorNpuFormat(output_tensor),
      result_type,
      output_size);
  at::Tensor result_cast = result_type == high_type ? result : NPUNativeFunctions::npu_dtype_cast(result, high_type);
  if (!NpuUtils::check_match(&result_cast)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result_cast);
    true_div_out_npu_nocheck(self_temp, other_temp, contiguous_result);
    NpuUtils::format_fresh_view(result_cast, contiguous_result);
  } else {
    true_div_out_npu_nocheck(self_temp, other_temp, result_cast);
  }

  if (result_type != high_type) {
    result_cast = NPUNativeFunctions::npu_dtype_cast(result_cast, result_type);
    result.copy_(result_cast);
  }
  return result;
}

at::Tensor NPUNativeFunctions::true_divide(const at::Tensor& self, const at::Tensor& other) {
  auto high_type = get_divide_high_type(self, other);
  at::Tensor self_temp = (self.scalar_type() == high_type) ? self : self.to(high_type);
  at::Tensor other_temp = (other.scalar_type() == high_type) ? other : other.to(high_type);

  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self_temp) || OpPreparation::IsCPUScalar(self_temp);
  at::Tensor output_tensor = is_self_wrapped ? other_temp : self_temp;
  auto output_size = broadcast_ops_npu_output_size(self_temp, other_temp);

  at::Tensor result = OpPreparation::ApplyTensor(output_tensor, output_size);
  true_div_out_npu_nocheck(self_temp, other_temp, result);

  return result;
}

at::Tensor NPUNativeFunctions::true_divide(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  true_div_scalar_out_nocheck(self, other, result);
  return result;
}

at::Tensor& NPUNativeFunctions::true_divide_(at::Tensor& self, const at::Tensor& other) {
  return NPUNativeFunctions::true_divide_out(self, other, self);
}

at::Tensor& NPUNativeFunctions::true_divide_(at::Tensor& self, const at::Scalar& other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    true_div_scalar_out_nocheck(contiguous_self, other, contiguous_self);
    NpuUtils::format_fresh_view(self, contiguous_self);
  } else {
    true_div_scalar_out_nocheck(self, other, self);
  }
  return self;
}
} // namespace native
} // namespace at_npu
