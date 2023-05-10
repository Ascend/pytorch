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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::expm1_out(const at::Tensor& self, at::Tensor& out) {
  OpCommand cmd;
  cmd.Name("Expm1")
      .Input(self)
      .Output(out)
      .Run();
  return out;
}

at::Tensor NPUNativeFunctions::expm1(const at::Tensor& self) {
  auto outputSize = input_same_output_size(self); 
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  NPUNativeFunctions::expm1_out(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::expm1_(at::Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = NPUNativeFunctions::expm1_out(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::expm1_out(self, self);
  }
  return self;
}
}  // namespace native
}  // namespace at_npu
