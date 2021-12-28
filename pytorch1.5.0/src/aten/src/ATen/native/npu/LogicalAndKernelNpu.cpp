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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& logical_and_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Scalar other) {
    auto selfCopy = (self.dtype() == at::kBool) ?
        self : self.npu_dtype_cast(at::kBool);
    OpCommand cmd;
    cmd.Name("LogicalAnd")
        .Input(selfCopy)
        .Input(other, selfCopy.scalar_type())
        .Output(result)
        .Run();

  return result;
}

Tensor& logical_and_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
    if (self.dim() == 0) {
      logical_and_out_npu_nocheck(result, other, self.item());
    } else if (other.dim() == 0) {
      logical_and_out_npu_nocheck(result, self, other.item());
    } else {
      auto selfCopy = (self.dtype() == at::kBool) ?
        self : self.npu_dtype_cast(at::kBool);

      auto otherCopy = (other.dtype() == at::kBool) ?
        other : other.npu_dtype_cast(at::kBool);

      OpCommand cmd;
      cmd.Name("LogicalAnd")
        .Input(selfCopy)
        .Input(otherCopy)
        .Output(result)
        .Run();
    }    

  return result;
}

Tensor& logical_and_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      result.scalar_type(),
      outputSize);
  
  if (NpuUtils::check_match(&result) && (result.dtype() == at::kBool)) {
    logical_and_out_npu_nocheck(result, self, other);
  } else {
    auto resultCopy = OpPreparation::ApplyTensorWithSizes(
        outputSize, self.options().dtype(at::kBool));

    logical_and_out_npu_nocheck(resultCopy, self, other);

    resultCopy = resultCopy.npu_dtype_cast(self.scalar_type());
    NpuUtils::format_fresh_view(result, resultCopy);
  }

  return result;
}

Tensor logical_and_npu(const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kBool), 
      CalcuOpUtil::get_tensor_npu_format(self));

  logical_and_out_npu_nocheck(result, self, other);

  return result;
}

Tensor& logical_and_npu_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.dtype() == other.dtype(),
      "Expected object of scalar type ", self.dtype(),
      " but got scalar type ", other.dtype(), " for argument 'other'");

  SmallVector<Tensor, N> inputs = {self, other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (NpuUtils::check_match(&self) && (self.dtype() == at::kBool)) {
    logical_and_out_npu_nocheck(self, self, other);
  } else {
    auto outputSize = broadcast_ops_npu_output_size(self, other);
    auto result = OpPreparation::ApplyTensorWithSizes(
        outputSize, self.options().dtype(at::kBool));

    logical_and_out_npu_nocheck(result, self, other);

    result = result.npu_dtype_cast(self.scalar_type());
    NpuUtils::format_fresh_view(self, result);
  }

  return self;
}

} // namespace native
} // namespace at