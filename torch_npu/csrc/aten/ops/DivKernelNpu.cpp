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
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

void check_rounding_mode(c10::optional<c10::string_view> rounding_mode) {
  TORCH_CHECK((!rounding_mode.has_value() || *rounding_mode == "trunc" || *rounding_mode == "floor"),
      "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
      "but found '", *rounding_mode, "'");
}

std::vector<at::Tensor> compute_complex_coefficient(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor other_copy = other.sizes() == at::IntArrayRef{} ?
      CalcuOpUtil::CopyScalarToDevice(other.item(), other.scalar_type()) : other;

  c10::SmallVector<at::Tensor, N> self_splite = complex_compute_pre_check_split(self);
  c10::SmallVector<at::Tensor, N> other_splite = complex_compute_pre_check_split(other_copy);

  auto ac = self_splite[0].mul(other_splite[0]);
  auto bc = self_splite[1].mul(other_splite[0]);
  auto ad = self_splite[0].mul(other_splite[1]);
  auto bd = self_splite[1].mul(other_splite[1]);
  auto down = other_splite[0].mul(other_splite[0]).add(other_splite[1].mul(other_splite[1]));

  std::vector<at::Tensor> final_coeffs;
  final_coeffs.push_back(ac.add(bd));
  final_coeffs.push_back(bc.sub(ad));
  final_coeffs.push_back(down);
  return final_coeffs;
}

at::Tensor& div_scalar_out_npu(const at::Tensor& self, const at::Scalar other, at::Tensor& result) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("RealDiv")
      .Expect(unified_result)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& div_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  if (OpPreparation::IsCPUScalar(other)) {
    div_scalar_out_npu(self, other.item(), result);
  } else {
    auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
    OpCommand cmd;
    cmd.Name("RealDiv")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}

at::Tensor& compute_div_out_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor output_tensor = CalcuOpUtil::IsScalarWrappedToTensor(self) ? other : self;
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  if (isFloatingType(result.scalar_type())) {
    high_type = result.scalar_type();
  }

  TORCH_CHECK(canCast(high_type, result.scalar_type()),
      "result type ", high_type, " can't be cast to the desired output type ", result.scalar_type());

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(output_tensor),
      high_type,
      output_size);
  at::Tensor self_copy = (self.scalar_type() != high_type && !CalcuOpUtil::IsScalarWrappedToTensor(self) &&
      at_npu::key::isDeviceTensor(self)) ? NPUNativeFunctions::npu_dtype_cast(self, high_type) : self;
  at::Tensor other_copy = (other.scalar_type() != high_type && !CalcuOpUtil::IsScalarWrappedToTensor(other) &&
      at_npu::key::isDeviceTensor(other)) ? NPUNativeFunctions::npu_dtype_cast(other, high_type) : other;
  div_out_npu_nocheck(self_copy, other_copy, result);

  return result;
}

at::Tensor& NPUNativeFunctions::div_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  if (self.is_complex() || other.is_complex()) {
    if (self.scalar_type() != c10::ScalarType::ComplexDouble &&
        other.scalar_type() != c10::ScalarType::ComplexDouble) {
      std::vector<at::Tensor> coeffs = compute_complex_coefficient(self, other);
      at::Tensor left_result = OpPreparation::ApplyTensor(result, coeffs[0].options());
      at::Tensor right_result = OpPreparation::ApplyTensor(result, coeffs[1].options());
      compute_div_out_nocheck(coeffs[0], coeffs[2], left_result);
      compute_div_out_nocheck(coeffs[1], coeffs[2], right_result);
      auto final_result = NPUNativeFunctions::stack({left_result, right_result}, -1);
      result.copy_(at::native::view_as_complex(final_result));
      return result;
    }
  }
  compute_div_out_nocheck(self, other, result);
  return result;
}

at::Tensor& compute_div_out_mode_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode,
    at::Tensor& result) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    NPUNativeFunctions::floor_divide_out(self, other, result);
    return result;
  }

  if (!rounding_mode.has_value()) {
    NPUNativeFunctions::div_out(self, other, result);
    return result;
  } else if (*rounding_mode == "trunc") {
    at::ScalarType high_dtype = at::native::result_type(self, other);
    at::ScalarType result_dtype = result.scalar_type();

    TORCH_CHECK(canCast(high_dtype, result_dtype),
        "result type ", high_dtype, " can't be cast to the desired output type ", result_dtype);

    if (isFloatingType(result_dtype)) {
      NPUNativeFunctions::div_out(self, other, result);
      NPUNativeFunctions::trunc_(result);
    } else {
      at::Tensor result_cast = NPUNativeFunctions::div(self, other);
      NPUNativeFunctions::trunc_(result_cast);
      if (result_cast.scalar_type() != result_dtype) {
        result_cast = NPUNativeFunctions::npu_dtype_cast(result_cast, result_dtype);
      }
      result.copy_(result_cast);
    }
    return result;
  }
}

at::Tensor& NPUNativeFunctions::div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode,
    at::Tensor& result) {
  check_rounding_mode(rounding_mode);
  if (self.is_complex() || other.is_complex()) {
    if (self.scalar_type() != c10::ScalarType::ComplexDouble &&
        other.scalar_type() != c10::ScalarType::ComplexDouble) {
      std::vector<at::Tensor> coeffs = compute_complex_coefficient(self, other);
      at::Tensor output_tensor =
          CalcuOpUtil::IsScalarWrappedToTensor(coeffs[0]) ? coeffs[2] : coeffs[0];
      auto output_size = broadcast_ops_npu_output_size(coeffs[0], coeffs[2]);
      at::ScalarType high_type = at::native::result_type(coeffs[0], coeffs[2]);
      if (isIntegralType(high_type, true)) {
        high_type = at::ScalarType::Float;
      }
      at::Tensor left_result = OpPreparation::ApplyTensorWithFormat(
          output_size,
          output_tensor.options().dtype(high_type),
          CalcuOpUtil::GetTensorNpuFormat(output_tensor));
      at::Tensor right_result = OpPreparation::ApplyTensorWithFormat(
          output_size,
          output_tensor.options().dtype(high_type),
          CalcuOpUtil::GetTensorNpuFormat(output_tensor));
      compute_div_out_mode_nocheck(coeffs[0], coeffs[2], rounding_mode, left_result);
      compute_div_out_mode_nocheck(coeffs[1], coeffs[2], rounding_mode, right_result);
      auto final_result = NPUNativeFunctions::stack({left_result, right_result}, -1);
      result.copy_(at::native::view_as_complex(final_result));
      return result;
    }
  }
  compute_div_out_mode_nocheck(self, other, rounding_mode, result);
  return result;
}

at::Tensor compute_div_nocheck(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  at::Tensor output_tensor = is_self_wrapped ? other : self;

  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  at::Tensor self_copy = (self.scalar_type() != high_type && !CalcuOpUtil::IsScalarWrappedToTensor(self) &&
      at_npu::key::isDeviceTensor(self)) ? NPUNativeFunctions::npu_dtype_cast(self, high_type) : self;
  at::Tensor other_copy = (other.scalar_type() != high_type && !CalcuOpUtil::IsScalarWrappedToTensor(other) &&
      at_npu::key::isDeviceTensor(other)) ? NPUNativeFunctions::npu_dtype_cast(other, high_type) : other;

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      output_size,
      output_tensor.options().dtype(high_type),
      CalcuOpUtil::GetTensorNpuFormat(output_tensor));

  div_out_npu_nocheck(self_copy, other_copy, result);
  return result;
}

at::Tensor compute_div_complex_nocheck(const at::Tensor& self, const at::Tensor& other) {
  std::vector<at::Tensor> coeffs = compute_complex_coefficient(self, other);
  at::Tensor left_result = compute_div_nocheck(coeffs[0], coeffs[2]);
  at::Tensor right_result = compute_div_nocheck(coeffs[1], coeffs[2]);
  auto final_result = NPUNativeFunctions::stack({left_result, right_result}, -1);
  auto complex_result = at::native::view_as_complex(final_result);
  return complex_result;
}

at::Tensor NPUNativeFunctions::div(const at::Tensor& self, const at::Tensor& other) {
  if (self.is_complex() || other.is_complex()) {
    if (self.scalar_type() != c10::ScalarType::ComplexDouble &&
        other.scalar_type() != c10::ScalarType::ComplexDouble) {
      return compute_div_complex_nocheck(self, other);
    }
  }
  return compute_div_nocheck(self, other);
}

at::Tensor NPUNativeFunctions::div(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  div_scalar_out_npu(self, other, result);
  return result;
}

at::Tensor NPUNativeFunctions::div(
    const at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  check_rounding_mode(rounding_mode);
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return NPUNativeFunctions::floor_divide(self, other);
  }

  at::Tensor true_div_res = NPUNativeFunctions::div(self, other);
  if (!rounding_mode.has_value()) {
    return true_div_res;
  } else if (*rounding_mode == "trunc") {
    return NPUNativeFunctions::trunc(true_div_res);
  }
}

at::Tensor compute_div_mode_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return NPUNativeFunctions::floor_divide(self, other);
  }
  at::Tensor div_res = NPUNativeFunctions::div(self, other);
  if (!rounding_mode.has_value()) {
    return div_res;
  } else if (*rounding_mode == "trunc") {
    at::Tensor trunc_div_res = NPUNativeFunctions::trunc(div_res);
    at::ScalarType high_dtype = at::native::result_type(self, other);
    if (trunc_div_res.scalar_type() != high_dtype) {
      trunc_div_res = NPUNativeFunctions::npu_dtype_cast(trunc_div_res, high_dtype);
    }
    return trunc_div_res;
  }
}

at::Tensor compute_div_mode_complex_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  std::vector<at::Tensor> coeffs = compute_complex_coefficient(self, other);
  at::Tensor left_result = compute_div_mode_nocheck(coeffs[0], coeffs[2], rounding_mode);
  at::Tensor right_result = compute_div_mode_nocheck(coeffs[1], coeffs[2], rounding_mode);
  auto final_result = NPUNativeFunctions::stack({left_result, right_result}, -1);
  auto complex_result = at::native::view_as_complex(final_result);
  return complex_result;
}

at::Tensor NPUNativeFunctions::div(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  check_rounding_mode(rounding_mode);
  if (self.is_complex() || other.is_complex()) {
    if (self.scalar_type() != c10::ScalarType::ComplexDouble &&
        other.scalar_type() != c10::ScalarType::ComplexDouble) {
      return compute_div_mode_complex_nocheck(self, other, rounding_mode);
    }
  }
  return compute_div_mode_nocheck(self, other, rounding_mode);
}

at::Tensor& compute_div__nocheck(at::Tensor& self, const at::Tensor& other) {
  c10::SmallVector<at::Tensor, N> inputs = {self, other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    NPUNativeFunctions::div_out(contiguous_self, other, contiguous_self);
    NpuUtils::format_fresh_view(self, contiguous_self);
  } else {
    div_out_npu_nocheck(self, other, self);
  }
  return self;
}

at::Tensor& compute_div__complex_nocheck(at::Tensor& self, const at::Tensor& other) {
  std::vector<at::Tensor> coeffs = compute_complex_coefficient(self, other);
  at::Tensor& left_result = compute_div__nocheck(coeffs[0], coeffs[2]);
  at::Tensor& right_result = compute_div__nocheck(coeffs[1], coeffs[2]);
  auto final_result = NPUNativeFunctions::stack({left_result, right_result}, -1);
  self = at::native::view_as_complex(final_result);
  return self;
}

at::Tensor& NPUNativeFunctions::div_(at::Tensor& self, const at::Tensor& other) {
  if (self.is_complex() || other.is_complex()) {
    if (self.scalar_type() != c10::ScalarType::ComplexDouble &&
        other.scalar_type() != c10::ScalarType::ComplexDouble) {
      return compute_div__complex_nocheck(self, other);
    }
  }
  return compute_div__nocheck(self, other);
}

at::Tensor& NPUNativeFunctions::div_(at::Tensor& self, const at::Scalar& other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    div_scalar_out_npu(contiguous_self, other, contiguous_self);
    NpuUtils::format_fresh_view(self, contiguous_self);
  } else {
    div_scalar_out_npu(self, other, self);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::div_(
    at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  check_rounding_mode(rounding_mode);
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return NPUNativeFunctions::floor_divide_(self, other);
  }
  NPUNativeFunctions::div_(self, other);
  if (!rounding_mode.has_value()) {
    return self;
  } else if (*rounding_mode == "trunc") {
    return NPUNativeFunctions::trunc_(self);
  }
}

at::Tensor& compute_div__mode_nocheck(
    at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return NPUNativeFunctions::floor_divide_(self, other);
  }
  NPUNativeFunctions::div_(self, other);
  if (!rounding_mode.has_value()) {
    return self;
  } else if (*rounding_mode == "trunc") {
    return NPUNativeFunctions::trunc_(self);
  }
}

at::Tensor& compute_div__mode_complex_nocheck(
    at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  std::vector<at::Tensor> coeffs = compute_complex_coefficient(self, other);
  at::Tensor& left_result = compute_div__mode_nocheck(coeffs[0], coeffs[2], rounding_mode);
  at::Tensor& right_result = compute_div__mode_nocheck(coeffs[1], coeffs[2], rounding_mode);
  auto final_result = NPUNativeFunctions::stack({left_result, right_result}, -1);
  self = at::native::view_as_complex(final_result);
  return self;
}

at::Tensor& NPUNativeFunctions::div_(
    at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  check_rounding_mode(rounding_mode);
  if (self.is_complex() || other.is_complex()) {
    if (self.scalar_type() != c10::ScalarType::ComplexDouble &&
        other.scalar_type() != c10::ScalarType::ComplexDouble) {
      return compute_div__mode_complex_nocheck(self, other, rounding_mode);
    }
  }
  return compute_div__mode_nocheck(self, other, rounding_mode);
}

} // namespace native
} // namespace at_npu
