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

at::Tensor& NPUNativeFunctions::hardtanh_out(
    const at::Tensor& self,
    const at::Scalar& min,
    const at::Scalar& max,
    at::Tensor& result) {
  OpPreparation::CheckMemory({self}, {result});
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(min, self.scalar_type())
      .Input(max, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::hardtanh(const at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  hardtanh_out(self, min, max, result);
  return result;
}

at::Tensor& NPUNativeFunctions::hardtanh_(at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = hardtanh_out(contiguousSelf, min, max, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    hardtanh_out(self, min, max, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu
