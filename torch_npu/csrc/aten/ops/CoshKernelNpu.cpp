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
 at::Tensor& NPUNativeFunctions::cosh_out(const  at::Tensor& self,  at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Cosh")
     .Input(self)
     .Output(result)
     .Run();

  return result;
}

 at::Tensor& NPUNativeFunctions::cosh_( at::Tensor& self) {
  OpPreparation::CheckMemory({self}, {self});

  if (!NpuUtils::check_match(&self)) {
     at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
     at::Tensor result = NPUNativeFunctions::cosh_out(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::cosh_out(self, self);
  }

  return self;
}

 at::Tensor NPUNativeFunctions::cosh(const  at::Tensor& self) {
   at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  NPUNativeFunctions::cosh_out(self, result);
  return result;
}

} // namespace native
} // namespace at_npu