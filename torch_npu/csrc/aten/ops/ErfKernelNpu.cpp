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

at::Tensor& erf_npu_nocheck(const at::Tensor& self, at::Tensor& out) {
  OpCommand cmd;
  cmd.Name("Erf")
    .Input(self)
    .Output(out)
    .Run();
  return out;
}

at::Tensor& NPUNativeFunctions::erf_out(const at::Tensor& self, at::Tensor& out) {
  OpPreparation::CheckOut(
      {self},
      out,
      self);

  if (!NpuUtils::check_match(&out)) {
      at::Tensor contiguousResult = NpuUtils::format_contiguous(out);
      at::Tensor newResult = erf_npu_nocheck(self, contiguousResult);
      NpuUtils::format_fresh_view(out, newResult);
  } else {
      erf_npu_nocheck(self, out);
  }
  return out;
}

at::Tensor NPUNativeFunctions::erf(const at::Tensor& self) {
  auto outputSize = input_same_output_size(self); 
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  erf_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::erf_(at::Tensor& self) {
  NPUNativeFunctions::erf_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu