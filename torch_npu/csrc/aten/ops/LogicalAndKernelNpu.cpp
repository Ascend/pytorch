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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor& logical_and_out_npu_nocheck(
    const at::Tensor& self,
    const at::Scalar other,
    at::Tensor& result) {
    auto selfCopy = (self.dtype() == at::kBool) ?
        self : NPUNativeFunctions::npu_dtype_cast(self, at::kBool);
    OpCommand cmd;
    cmd.Name("LogicalAnd")
        .Input(selfCopy)
        .Input(other, selfCopy.scalar_type())
        .Output(result)
        .Run();
  return result;
}

at::Tensor& logical_and_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  if (self.dim() == 0) {
    logical_and_out_npu_nocheck(other, self.item(), result);
  } else if (other.dim() == 0) {
    logical_and_out_npu_nocheck(self, other.item(), result);
  } else {
    auto selfCopy = (self.dtype() == at::kBool) ? 
        self : NPUNativeFunctions::npu_dtype_cast(self, at::kBool);
    auto otherCopy = (other.dtype() == at::kBool) ?
        other : NPUNativeFunctions::npu_dtype_cast(other, at::kBool);

    OpCommand cmd;
    cmd.Name("LogicalAnd")
      .Input(selfCopy)
      .Input(otherCopy)
      .Output(result)
      .Run();
    }    
  return result;
}

at::Tensor& NPUNativeFunctions::logical_and_out(
    const at::Tensor& self, 
    const at::Tensor& other,
    at::Tensor& result) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      result.scalar_type(),
      outputSize);
  if (NpuUtils::check_match(&result) && (result.dtype() == at::kBool)) {
    logical_and_out_npu_nocheck(self, other, result);
  } else {
    auto resultCopy = OpPreparation::ApplyTensorWithSizes(
        outputSize, self.options().dtype(at::kBool));
    logical_and_out_npu_nocheck(self, other, resultCopy);
    resultCopy = NPUNativeFunctions::npu_dtype_cast(resultCopy, self.scalar_type());
    NpuUtils::format_fresh_view(result, resultCopy);
  }
  return result;
}

at::Tensor NPUNativeFunctions::logical_and(const at::Tensor& self, const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kBool), 
      CalcuOpUtil::get_tensor_npu_format(self));
  logical_and_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor& NPUNativeFunctions::logical_and_(at::Tensor& self, const at::Tensor& other) {
  TORCH_CHECK(
      self.dtype() == other.dtype(),
      "Expected object of scalar type ", self.dtype(),
      " but got scalar type ", other.dtype(), " for argument 'other'");
  OpPreparation::CheckMemory({self, other}, {self});
  logical_and_out(self, other, self);
  return self;
}
} // namespace native
} // namespace at_npu
