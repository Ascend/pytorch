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

at::Tensor& asin_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Asin")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::asin_out(
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  asin_out_npu_nocheck(result, self);
  return result;
}

at::Tensor NPUNativeFunctions::asin(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  asin_out_npu_nocheck(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::asin_(at::Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = asin_out_npu_nocheck(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    asin_out_npu_nocheck(self, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu
