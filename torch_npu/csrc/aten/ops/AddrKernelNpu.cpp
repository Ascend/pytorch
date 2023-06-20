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

#include <ATen/native/TypeProperties.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
void check_beta_aplha(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    at::Scalar beta,
    at::Scalar alpha,
    at::ScalarType high_dtype) {
  TORCH_CHECK(((high_dtype == at::ScalarType::Bool) || !beta.isBoolean()),
      "Boolean beta only supported for Boolean results.");
  TORCH_CHECK(((high_dtype == at::ScalarType::Bool) || !alpha.isBoolean()),
      "Boolean alpha only supported for Boolean results.");

  bool all_int_inputs = ((isIntegralType(self.scalar_type(), true)) && (isIntegralType(vec1.scalar_type(), true)) &&
      (isIntegralType(vec2.scalar_type(), true)));

  TORCH_CHECK(!all_int_inputs || beta.isIntegral(true),
      "For integral input tensors, argument beta must not be a floating point number.");
  TORCH_CHECK(!all_int_inputs || alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating point number.");
}

at::Tensor& NPUNativeFunctions::addr_out(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    at::Scalar beta,
    at::Scalar alpha,
    at::Tensor& result) {
  at::ScalarType high_dtype = at::native::result_type({self, vec1, vec2});
  check_beta_aplha(self, vec1, vec2, beta, alpha, high_dtype);
  NpuUtils::check_1d(vec1, "vec1", "addr");
  NpuUtils::check_1d(vec2, "vec2", "addr");

  bool result_to_cast = (high_dtype == at::ScalarType::Bool);

  at::Tensor self_cast = result_to_cast ? NPUNativeFunctions::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor vec1_cast = result_to_cast ? NPUNativeFunctions::npu_dtype_cast(vec1, at::kFloat) : vec1;
  at::Tensor vec2_cast = result_to_cast ? NPUNativeFunctions::npu_dtype_cast(vec2, at::kFloat) : vec2;
  at::Tensor result_cast = result_to_cast ? NPUNativeFunctions::npu_dtype_cast(result, at::kFloat) : result;
  at::Scalar beta_cast = result_to_cast ? beta.toFloat() : beta;
  at::Scalar alpha_cast = result_to_cast ? alpha.toFloat() : alpha;

  at::Tensor mul1 = vec1_cast.unsqueeze(1);
  at::Tensor mul2 = vec2_cast.unsqueeze(0);

  // vecmul vec1&vec2
  at::Tensor mul_result = at::mul(mul1, mul2);

  // mul*alpha
  at::Tensor mul_result_alpha = at::mul(mul_result, alpha_cast);

  // mul*alpha+self*beta
  at::add_out(result_cast, mul_result_alpha, self_cast, beta_cast);

  if (result_to_cast) {
    result_cast = NPUNativeFunctions::npu_dtype_cast(result_cast, at::ScalarType::Bool);
    result.copy_(result_cast);
  }

  return result;
}

at::Tensor NPUNativeFunctions::addr(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    at::Scalar beta,
    at::Scalar alpha) {
  at::ScalarType high_dtype = at::native::result_type({self, vec1, vec2});
  auto output_size = addr_npu_output_size(self, vec1, vec2, beta, alpha);
  at::Tensor result = OpPreparation::ApplyTensor(output_size, self.options().dtype(high_dtype), self);
  addr_out(self, vec1, vec2, beta, alpha, result);

  return result;
}

at::Tensor& NPUNativeFunctions::addr_(
    at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    at::Scalar beta,
    at::Scalar alpha) {
  OpPreparation::CheckMemory({self, vec1, vec2}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    at::Tensor result =
        addr_out(contiguous_self, vec1, vec2, beta, alpha, contiguous_self);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addr_out(self, vec1, vec2, beta, alpha, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu
