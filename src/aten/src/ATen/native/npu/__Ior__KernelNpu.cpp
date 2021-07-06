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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& __ior___out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  string real_op_name = (self.dtype() == ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

Tensor& __ior___out_npu(Tensor& result, const Tensor& self, Scalar other) {
  string real_op_name = (self.dtype() == ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

Tensor& __ior___npu(Tensor& self, const Tensor& other) { 
  OpPreparation::CheckMemory({self, other}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = __ior___out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    __ior___out_npu(self, self, other);
  }

  return self;
}

Tensor& __ior___npu(Tensor& self, Scalar other) {     
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = __ior___out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    __ior___out_npu(self, self, other);
  }

  return self;
}

}
} // namespace at::native
