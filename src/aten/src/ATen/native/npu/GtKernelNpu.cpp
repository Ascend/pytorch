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

Tensor& gt_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& other) {
  auto unified_result = OpPreparation::comparison_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("Greater")
     .Expect(unified_result)
     .Input(self)
     .Input(other)
     .Output(result)
     .Run();

  return result;
}

Tensor& gt_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf),
      result.scalar_type(),
      outputSize);

  gt_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor& gt_out_npu_nocheck(Tensor& result, const Tensor& self, Scalar other) {
  OpCommand cmd;
  cmd.Name("Greater")
     .Input(self)
     .Input(other, self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

Tensor& gt_out_npu(Tensor& result, const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes(); 
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf),
      result.scalar_type(),
      outputSize);

  gt_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor gt_npu(const Tensor& self, const Tensor& other) {
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
  gt_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

Tensor gt_npu(const Tensor& self, Scalar other) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfSelf);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      formatCastOfSelf.options().dtype(kBool),
      ACL_FORMAT_ND);

  // calculate the output resugt of the NPU
  gt_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

Tensor& gt_npu_(Tensor& self, const Tensor& other) {
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
    gt_out_npu_nocheck(result, contiguousSelf, ori_other);
  } else {
    gt_out_npu_nocheck(result, self, ori_other);
  }

  // uint8 to self dtype
  self.copy_(result);

  return self;
}

Tensor& gt_npu_(Tensor& self, Scalar other) {
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
    gt_out_npu_nocheck(result, contiguousSelf, other);
  } else {
    gt_out_npu_nocheck(result, self, other);
  }

  // uint8 to self dtype
  self.copy_(result);

  return self;
}

} // namespace native
} // namespace at
