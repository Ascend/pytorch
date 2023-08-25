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
  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self) || OpPreparation::IsCPUScalar(self);
  at::Tensor output_tensor = is_self_wrapped ? other : self;

  auto result_type = result.scalar_type();
  auto output_size = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self, other},
      result,
      CalcuOpUtil::GetTensorNpuFormat(output_tensor),
      result_type,
      output_size);

  auto high_type = at::native::result_type(self, other);
  TORCH_CHECK(canCast(high_type, result_type), "result type ", high_type,
      " can't be cast to the desired output type ", result_type);

  if (high_type == at::kBool) {
    high_type = at::kFloat;
  }
  at::Tensor self_cast = (self.scalar_type() == high_type) ? self : self.to(high_type);
  at::Tensor other_cast = (other.scalar_type() == high_type) ? other : other.to(high_type);

  at::Tensor result_cast = (result_type != high_type) ?
      NPUNativeFunctions::npu_dtype_cast(result, high_type) : result;

  if (!NpuUtils::check_match(&result_cast)) {
    at::Tensor contiguous_result_cast = NpuUtils::format_contiguous(result_cast);
    mul_out_npu_nocheck(contiguous_result_cast, self_cast, other_cast);
    NpuUtils::format_fresh_view(result_cast, contiguous_result_cast);
  } else {
    mul_out_npu_nocheck(result_cast, self_cast, other_cast);
  }

  if (result_type != high_type) {
    result_cast = NPUNativeFunctions::npu_dtype_cast(result_cast, result_type);
    result.copy_(result_cast);
  }
  return result;
}

at::Tensor NPUNativeFunctions::mul(const at::Tensor& self, const at::Tensor& other) {
  auto high_type = at::native::result_type(self, other);
  bool out_is_bool = (high_type == at::kBool);
  if (out_is_bool) {
    high_type = at::kFloat;
  }

  at::Tensor self_cast = (self.scalar_type() == high_type) ? self : self.to(high_type);
  at::Tensor other_cast = (other.scalar_type() == high_type) ? other : other.to(high_type);

  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self_cast) || OpPreparation::IsCPUScalar(self_cast);
  at::Tensor output_tensor = is_self_wrapped ? other_cast : self_cast;

  auto output_size = broadcast_ops_npu_output_size(self_cast, other_cast);
  at::Tensor result = OpPreparation::ApplyTensor(output_tensor, output_size);
  mul_out_npu_nocheck(result, self_cast, other_cast);

  if (out_is_bool) {
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
  return NPUNativeFunctions::mul_out(self, other, self);
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
