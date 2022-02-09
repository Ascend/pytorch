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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& le_out_npu_nocheck(const at::Tensor& self, at::Scalar other, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("LessEqual")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::le_out(const at::Tensor& self, at::Scalar other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
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

at::Tensor& le_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
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

at::Tensor& NPUNativeFunctions::le_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
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

at::Tensor NPUNativeFunctions::le(const at::Tensor& self, at::Scalar other) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      formatCastOfSelf.sizes(),
      formatCastOfSelf.options().dtype(at::kBool),
      ACL_FORMAT_ND);
  le_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

at::Tensor NPUNativeFunctions::le(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);

  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      formatCastOfSelf.options().dtype(at::kBool),
      ACL_FORMAT_ND);

  le_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor& NPUNativeFunctions::le_(at::Tensor& self, at::Scalar other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckMemory({self}, {self}); 
  at::Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(at::ScalarType::Byte));
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    le_out_npu_nocheck(contiguousSelf, other, result);
  } else {
    le_out_npu_nocheck(self, other, result);
  }
  self.copy_(result);
  return self;
}

at::Tensor& NPUNativeFunctions::le_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  at::Tensor ori_other = OpPreparation::CastBackToOriFormat(other);
  OpPreparation::CheckMemory({self, ori_other}, {self}); 
  at::Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(at::ScalarType::Byte));
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    le_out_npu_nocheck(contiguousSelf, ori_other, result);
  } else {
    le_out_npu_nocheck(self, ori_other, result);
  }
  self.copy_(result);
  return self;
}

} // namespace native
} // namespace at_npu
