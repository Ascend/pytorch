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

at::Tensor& ilshift_out_npu(
    at::Tensor& result,
    at::Tensor& self,
    at::Scalar other) {
  at::Tensor otherBroadcast = OpPreparation::ApplyTensor(self).fill_(other);
  OpCommand cmd;
  cmd.Name("LeftShift")
     .Input(self)
     .Input(otherBroadcast)
     .Output(result)
     .Run();

  return result;
}

at::Tensor& ilshift_out_npu(
    at::Tensor& result,
    at::Tensor& self,
    const at::Tensor& other) {
    at::Tensor otherBroadcast = other.expand(self.sizes());
    OpCommand cmd;
    cmd.Name("LeftShift")
       .Input(self)
       .Input(otherBroadcast)
       .Output(result)
       .Run(); 

  return result;
}

at::Tensor& NPUNativeFunctions::__ilshift__(at::Tensor& self, const at::Tensor& other) {
  if(!NpuUtils::check_match(&self)){
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ilshift_out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    ilshift_out_npu(self, self, other);
  }

  return self;
}

at::Tensor& NPUNativeFunctions::__ilshift__(at::Tensor& self, at::Scalar other) {
  if(!NpuUtils::check_match(&self)){
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ilshift_out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    ilshift_out_npu(self, self, other);
  }

  return self;
}

} // namespace native
} // namespace at_npu
