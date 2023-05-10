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

at::Tensor& ior_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& ior_out_npu_nocheck(const at::Tensor& self, at::Scalar other, at::Tensor& result) {
  string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::__ior__(at::Tensor& self, const at::Tensor& other) { 
  OpPreparation::CheckMemory({self, other}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = ior_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    ior_out_npu_nocheck(self, other, self);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::__ior__(at::Tensor& self, const at::Scalar& other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = ior_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    ior_out_npu_nocheck(self, other, self);
  }
  return self;
}
} // namespace native
} // namespace at_npu
