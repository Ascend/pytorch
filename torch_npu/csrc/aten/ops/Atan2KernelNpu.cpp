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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& atan2_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("Atan2")
     .Expect(unified_result)
     .Input(self)
     .Input(other)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::atan2_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor result = atan2_out_npu_nocheck(self, other, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    atan2_out_npu_nocheck(self, other, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::atan2(const at::Tensor& self, const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  atan2_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor& NPUNativeFunctions::atan2_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CheckMemory({self, other}, {self});
  NPUNativeFunctions::atan2_out(self, other, self);
  return self;
}

} // namespace native
} // namespace at_npu
