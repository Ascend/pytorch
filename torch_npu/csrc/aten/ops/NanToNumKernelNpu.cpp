// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

const float DOUBLE_MAX_VALUE = 1.7976931348623157e+308;
const float DOUBLE_MIN_VALUE = -1.7976931348623157e+308;
const float FLOAT32_MAX_VALUE = 3.4028235e+38;
const float FLOAT32_MIN_VALUE = -3.4028235e+38;
const float FLOAT16_MAX_VALUE = 65504.0;
const float FLOAT16_MIN_VALUE = -65504.0;
const float BFLOAT16_MAX_VALUE = 3.3895314e+38;
const float BFLOAT16_MIN_VALUE = -3.3895314e+38;
const float DEFAULT_NAN = 0.0;

std::tuple<float, float> get_posinf_and_neginf(
    at::ScalarType self_dtype,
    c10::optional<double> posinf,
    c10::optional<double> neginf) {
  float new_posinf, new_neginf;
  bool posinf_has_value = posinf.has_value();
  bool neginf_has_value = neginf.has_value();

  if (posinf_has_value && neginf_has_value) {
    new_posinf = posinf.value();
    new_neginf = neginf.value();
  } else {
    switch (self_dtype) {
      case at::ScalarType::Double:
        new_posinf = posinf_has_value ? posinf.value() : DOUBLE_MAX_VALUE;
        new_neginf = neginf_has_value ? neginf.value() : DOUBLE_MIN_VALUE;
        break;
      case at::ScalarType::Float:
        new_posinf = posinf_has_value ? posinf.value() : FLOAT32_MAX_VALUE;
        new_neginf = neginf_has_value ? neginf.value() : FLOAT32_MIN_VALUE;
        break;
      case at::ScalarType::Half:
        new_posinf = posinf_has_value ? posinf.value() : FLOAT16_MAX_VALUE;
        new_neginf = neginf_has_value ? neginf.value() : FLOAT16_MIN_VALUE;
        break;
      case at::ScalarType::BFloat16:
        new_posinf = posinf_has_value ? posinf.value() : BFLOAT16_MAX_VALUE;
        new_neginf = neginf_has_value ? neginf.value() : BFLOAT16_MIN_VALUE;
        break;
      default:
        new_posinf = posinf_has_value ? posinf.value() : FLOAT32_MAX_VALUE;
        new_neginf = neginf_has_value ? neginf.value() : FLOAT32_MIN_VALUE;
        break;
    }
  }
  return std::tie(new_posinf, new_neginf);
}

at::Tensor& nan_to_num_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  float new_nan = nan.has_value() ? nan.value() : DEFAULT_NAN;
  auto new_posinf_neginf = get_posinf_and_neginf(self.scalar_type(), pos_inf, neg_inf);
  OpCommand cmd;
  cmd.Name("NanToNum")
      .Input(self)
      .Output(result)
      .Attr("nan", new_nan)
      .Attr("posinf", std::get<0>(new_posinf_neginf))
      .Attr("neginf", std::get<1>(new_posinf_neginf))
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::nan_to_num_out(
    const at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf,
    at::Tensor& result) {
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "nan_to_num: dtype of out: ",
      result.scalar_type(),
      " should be same as input: ",
      self.scalar_type());

  if (isIntegralType(self.scalar_type(), true)) {
    result.resize_(self.sizes());
    result.copy_(self);
    return result;
  }

  OpPreparation::CheckOut(
      {self},
      result,
      self);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    nan_to_num_nocheck(contiguous_result, self, nan, pos_inf, neg_inf);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    nan_to_num_nocheck(result, self, nan, pos_inf, neg_inf);
  }
  return result;
}

at::Tensor NPUNativeFunctions::nan_to_num(
    const at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  auto result = OpPreparation::ApplyTensor(self);
  if (isIntegralType(self.scalar_type(), true)) {
    result.copy_(self);
    return result;
  }
  nan_to_num_nocheck(result, self, nan, pos_inf, neg_inf);
  return result;
}

at::Tensor& NPUNativeFunctions::nan_to_num_(
    at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  return nan_to_num_out(self, nan, pos_inf, neg_inf, self);
}

} // namespace native
} // namespace at_npu
