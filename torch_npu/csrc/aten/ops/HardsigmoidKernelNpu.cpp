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

at::Tensor& hardsigmoid_out_nocheck(
    const at::Tensor& self,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("HardSigmoid")
    .Input(self)
    .Output(result)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::hardsigmoid_out(
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor checkResult = hardsigmoid_out_nocheck(self, contiguousResult);
    NpuUtils::format_fresh_view(result, checkResult);
  } else {
    hardsigmoid_out_nocheck(self, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::hardsigmoid(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  hardsigmoid_out_nocheck(self, result);
  return result;
}

at::Tensor& hardsigmoid_(at::Tensor& self) {
  OpPreparation::CheckMemory({self}, {self});
  hardsigmoid_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
