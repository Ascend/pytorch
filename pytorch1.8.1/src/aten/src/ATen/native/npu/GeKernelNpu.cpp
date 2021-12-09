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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& ge_out_npu_nocheck(const Tensor& self, const Tensor& other, Tensor& result) {
  Tensor selfCast = self;
  Tensor otherCast = other;
  if (self.dtype() == ScalarType::Int || other.dtype() == ScalarType::Int 
      || self.dtype() == ScalarType::Bool || other.dtype() == ScalarType::Bool) {
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

Tensor& ge_out_npu(const Tensor& self, const Tensor& other, Tensor& result) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  ge_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

Tensor& ge_out_npu_nocheck(const Tensor& self, Scalar other, Tensor& result) {
  Tensor selfCast = self;
  if (self.dtype() == ScalarType::Int || self.dtype() == ScalarType::Bool) {
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

Tensor& ge_scalar_out_npu(const Tensor& self, Scalar other, Tensor& result) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes(); 
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  ge_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

Tensor ge_npu(const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      formatCastOfSelf.options().dtype(kBool),
      ACL_FORMAT_ND);
  ge_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

Tensor ge_scalar_npu(const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      formatCastOfSelf.sizes(),
      formatCastOfSelf.options().dtype(kBool),
      ACL_FORMAT_ND);
  ge_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

Tensor& ge_npu_(Tensor& self, const Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  Tensor ori_other = OpPreparation::CastBackToOriFormat(other);
  OpPreparation::CheckMemory({self, ori_other}, {self}); 

  Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(ScalarType::Byte));

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ge_out_npu_nocheck(contiguousSelf, ori_other, result);
  } else {
    ge_out_npu_nocheck(self, ori_other, result);
  }
  self.copy_(result);
  return self;
}

Tensor& ge_scalar_npu_(Tensor& self, Scalar other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckMemory({self}, {self}); 
  Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(ScalarType::Byte));
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ge_out_npu_nocheck(contiguousSelf, other, result);
  } else {
    ge_out_npu_nocheck(self, other, result);
  }
  self.copy_(result);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("ge.Scalar_out", TORCH_FN(ge_scalar_out_npu));
  m.impl("ge.Scalar", TORCH_FN(ge_scalar_npu));
  m.impl("ge.Tensor_out", TORCH_FN(ge_out_npu));
  m.impl("ge.Tensor", TORCH_FN(ge_npu));
  m.impl("ge_.Scalar", TORCH_FN(ge_scalar_npu_));
  m.impl("ge_.Tensor", TORCH_FN(ge_npu_));
}

} // namespace native
} // namespace at
