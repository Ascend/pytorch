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

Tensor& floor_divide_scalar_npu_nocheck(Tensor& result, const Tensor& self, Scalar other) {
  OpCommand cmd;
  cmd.Name("FloorDiv")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();
  return result;
}

Tensor& floor_divide_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes();
  OpPreparation::CheckOut(
      {self, other},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      result.scalar_type(),
      outputSize);

  if (other.dim() == 0) {
    floor_divide_scalar_npu_nocheck(result, self, other.item());
  } else {
    OpCommand cmd;
    cmd.Name("FloorDiv")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();    
  }
  return result;
}

Tensor& check_self_dtype_npu(Tensor& self){
  if (self.dtype() == ScalarType::Bool ||
      self.dtype() == ScalarType::Int) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }
  return self;
}

std::tuple<Tensor, Tensor> check_dtype_npu(Tensor& self, Tensor& other){
  if (self.dtype() == ScalarType::Bool ||
      self.dtype() == ScalarType::Int &&
      other.scalar_type() == ScalarType::Double) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }
  if (other.scalar_type() == ScalarType::Double) {
    other = other.to(ScalarType::Float);
  }
  if (other.scalar_type() == ScalarType::Long) {
    other = other.to(ScalarType::Int);
  }
  return std::tie(self, other);
}

Tensor& floor_divide_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  Tensor selfCast = self;
  Tensor otherCast = other;
  check_dtype_npu(selfCast, otherCast);
  floor_divide_out_npu_nocheck(result, selfCast, otherCast);
  return result;
}

Tensor floor_divide_npu(const Tensor& self, const Tensor& other) {
  Tensor selfCast = self;
  Tensor otherCast = other;
  check_dtype_npu(selfCast, otherCast);
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfCast);
  Tensor outputTensor = isSelfWrapped ? otherCast : selfCast;
  auto outputSize = broadcast_ops_npu_output_size(selfCast, otherCast);

  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(selfCast));
  floor_divide_out_npu_nocheck(result, selfCast, otherCast);
  return result;
}

Tensor floor_divide_npu(const Tensor& self, Scalar other) {
  Tensor selfCast = self;
  check_self_dtype_npu(selfCast);
  Tensor result = OpPreparation::ApplyTensor(selfCast);
  floor_divide_scalar_npu_nocheck(result, selfCast, other);
  return result;
}

Tensor& floor_divide_npu_(Tensor& self, const Tensor& other) {
  Tensor otherCast = other;
  check_dtype_npu(self, otherCast);
  SmallVector<Tensor, N> inputs = {self, otherCast};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = floor_divide_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    floor_divide_out_npu_nocheck(self, self, otherCast);
  }
  return self;
}

Tensor& floor_divide_npu_(Tensor& self, Scalar other) {
  check_self_dtype_npu(self);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    floor_divide_scalar_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    floor_divide_scalar_npu_nocheck(self, self, other);
  }
  return self;
}

} // namespace native
} // namespace at
