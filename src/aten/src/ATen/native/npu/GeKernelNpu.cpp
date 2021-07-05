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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& ge_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor selfCast = self;
  Tensor otherCast = other;
  if(self.dtype() == ScalarType::Int || other.dtype() == ScalarType::Int 
      || self.dtype() == ScalarType::Bool || other.dtype() == ScalarType::Bool){
    selfCast = self.to(ScalarType::Float);
    otherCast = other.to(ScalarType::Float);
  }
  auto unified_result = OpPreparation::comparison_op_check(result, selfCast, otherCast, true);
  OpCommand cmd;
  cmd.Name("GreaterEqual")
     .Expect(unified_result)
     .Input(selfCast)
     .Input(otherCast)
     .Output(result)
     .Run();
  
  return result;
}

Tensor& ge_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf),
      result.scalar_type(),
      outputSize);

  ge_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor& ge_out_npu_nocheck(Tensor& result, const Tensor& self, Scalar other) {
  Tensor selfCast = self;
  if(self.dtype() == ScalarType::Int || self.dtype() == ScalarType::Bool){
    selfCast = self.to(ScalarType::Float);
  }
  OpCommand cmd;
  cmd.Name("GreaterEqual")
     .Input(selfCast)
     .Input(other, selfCast.scalar_type())
     .Output(result)
     .Run();

  return result;
}

Tensor& ge_out_npu(Tensor& result, const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes(); 
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf),
      result.scalar_type(),
      outputSize);

  ge_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor ge_npu(const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      formatCastOfSelf.options().dtype(kBool),
      ACL_FORMAT_ND);

  // calculate the output result of the NPU
  ge_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor ge_npu(const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfSelf);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      formatCastOfSelf.options().dtype(kBool),
      ACL_FORMAT_ND);

  // calculate the output resugt of the NPU
  ge_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor& ge_npu_(Tensor& self, const Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  Tensor ori_other = OpPreparation::CastBackToOriFormat(other);
  SmallVector<Tensor, N> inputs = {self, ori_other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  Tensor result = at::empty_with_format(
      self.sizes(),
      self.options().dtype(ScalarType::Byte),
      CalcuOpUtil::get_tensor_npu_format(self));

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ge_out_npu_nocheck(result, contiguousSelf, ori_other);
  } else {
    ge_out_npu_nocheck(result, self, ori_other);
  }
  self.npu_dtype_cast_(result);
  return self;
}

Tensor& ge_npu_(Tensor& self, Scalar other) {
  OpPreparation::CastBackToOriFormat(self);
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  Tensor result = at::empty_with_format(
      self.sizes(),
      self.options().dtype(ScalarType::Byte),
      CalcuOpUtil::get_tensor_npu_format(self));

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ge_out_npu_nocheck(result, contiguousSelf, other);

  } else {
    ge_out_npu_nocheck(result, self, other);
  }
  self.npu_dtype_cast_(result);
  return self;
}

} // namespace native
} // namespace at
