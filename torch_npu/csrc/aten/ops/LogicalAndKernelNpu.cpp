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
#include "torch_npu/csrc/framework/utils/OpTemplate.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& logical_and_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("LogicalAnd")
    .Input(self)
    .Input(other)
    .Output(result)
    .Run();
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
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    logical_and_out_npu_nocheck(self, other, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    logical_and_out_npu_nocheck(self, other, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::logical_and(const at::Tensor& self, const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  logical_and_out_npu_nocheck(self, other, result);
  result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  return result;
}

at::Tensor& NPUNativeFunctions::logical_and_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CheckMemory({self, other}, {self});
  logical_and_out(self, other, self);
  return self;
}
} // namespace native
} // namespace at_npu
