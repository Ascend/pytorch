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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& cosh_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Cosh")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::cosh_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(result),
      self.scalar_type(),
      self.sizes());

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    cosh_out_npu_nocheck(contiguous_result, self);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    cosh_out_npu_nocheck(result, self);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::cosh_(at::Tensor& self) {
   return NPUNativeFunctions::cosh_out(self, self);
}

at::Tensor NPUNativeFunctions::cosh(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  cosh_out_npu_nocheck(result, self);
  return result;
}

} // namespace native
} // namespace at_npu
