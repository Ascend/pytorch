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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& cos_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Cos")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::cos_out(
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor checkResult = cos_out_npu_nocheck(contiguousResult, self);
    NpuUtils::format_fresh_view(result, checkResult);
  } else {
    cos_out_npu_nocheck(result, self);
  }
  return result;
}

at::Tensor NPUNativeFunctions::cos(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  cos_out_npu_nocheck(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::cos_(at::Tensor& self) {
  return cos_out(self, self);
}
} // namespace native
} // namespace at_npu