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

Tensor& le_out_npu_nocheck(const Tensor& self, Scalar other, Tensor& result) {
  OpCommand cmd;
  cmd.Name("LessEqual")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

Tensor& le_scalar_out_npu(const Tensor& self, Scalar other, Tensor& result) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  le_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

Tensor& le_out_npu_nocheck(const Tensor& self, const Tensor& other, Tensor& result) {
  auto unified_result = OpPreparation::comparison_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("LessEqual")
      .Expect(unified_result)
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();

  return result;
}

Tensor& le_out_npu(const Tensor& self, const Tensor& other, Tensor& result) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  le_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

Tensor le_scalar_npu(const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      formatCastOfSelf.sizes(),
      formatCastOfSelf.options().dtype(kBool),
      ACL_FORMAT_ND);
  le_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

Tensor le_npu(const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      formatCastOfSelf.options().dtype(kBool),
      ACL_FORMAT_ND);
  // calculate the output result of the NPU
  le_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

Tensor& le_scalar_npu_(Tensor& self, Scalar other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckMemory({self}, {self}); 
  Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(ScalarType::Byte));
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    le_out_npu_nocheck(contiguousSelf, other, result);
  } else {
    le_out_npu_nocheck(self, other, result);
  }
  self.copy_(result);
  return self;
}

Tensor& le_npu_(Tensor& self, const Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  Tensor ori_other = OpPreparation::CastBackToOriFormat(other);
  OpPreparation::CheckMemory({self, ori_other}, {self}); 
  Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(ScalarType::Byte));
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    le_out_npu_nocheck(contiguousSelf, ori_other, result);
  } else {
    le_out_npu_nocheck(self, ori_other, result);
  }
  self.copy_(result);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("le.Scalar_out", TORCH_FN(le_scalar_out_npu));
  m.impl("le.Scalar", TORCH_FN(le_scalar_npu));
  m.impl("le.Tensor_out", TORCH_FN(le_out_npu));
  m.impl("le.Tensor", TORCH_FN(le_npu));
  m.impl("le_.Scalar", TORCH_FN(le_scalar_npu_));
  m.impl("le_.Tensor", TORCH_FN(le_npu_));
}
} // namespace native
} // namespace at
