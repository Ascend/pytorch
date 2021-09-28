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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& __ilshift___out_npu(
    Tensor& result,
    Tensor& self,
    Scalar other) {
  // TODO: The op does not support the inconsistent shape of the two input
  Tensor otherBroadcast = at::empty(self.sizes(), self.options()).fill_(other); 
  OpCommand cmd;
  cmd.Name("LeftShift")
     .Input(self)
     .Input(otherBroadcast)
     .Output(result)
     .Run();

  return result;
}

Tensor& __ilshift___out_npu(
    Tensor& result,
    Tensor& self,
    const Tensor& other) {
    // TODO: The op does not support the inconsistent shape of the two input
    Tensor otherBroadcast = other.expand(self.sizes());
    OpCommand cmd;
    cmd.Name("LeftShift")
       .Input(self)
       .Input(otherBroadcast)
       .Output(result)
       .Run(); 

  return result;
}

Tensor& __iLshift___npu(Tensor& self, const Tensor& other) {
  OpPreparation::CheckMemory({self}, {self});  

  if(!NpuUtils::check_match(&self)){
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    __ilshift___out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    __ilshift___out_npu(self, self, other);
  }

  return self;
}

Tensor& __iLshift___npu(Tensor& self, Scalar other) {
  OpPreparation::CheckMemory({self}, {self});  

  if(!NpuUtils::check_match(&self)){
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    __ilshift___out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    __ilshift___out_npu(self, self, other);
  }

  return self;
}

} // namespace native
} // namespace at
