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
#include<ATen/NamedTensorUtils.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor scatter_add_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  OpCommand cmd;
  cmd.Name("ScatterAddWithAxis")
     .Input(self)
     .Input(index)
     .Input(src)
     .Output(result)
     .Attr("axis", dim)
     .Run();
  return result;
}

Tensor scatter_add_npu(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  return self.clone(at::MemoryFormat::Contiguous).scatter_add_(dim, index, src);
}

Tensor& scatter_add_npu_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  OpPreparation::CheckMemory({self, index, src}, {self});

  ScalarType selfType = self.scalar_type();
  Tensor selfFp32 = self;
  Tensor srcFp32 = src;
  if (self.scalar_type() == ScalarType::Half) {
    selfFp32 = self.to(ScalarType::Float);
    srcFp32 = src.to(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    Tensor result =
        scatter_add_out_npu(contiguousSelf, contiguousSelf, dim, index, srcFp32);
    self.copy_(result);
  } else {
    scatter_add_out_npu(selfFp32, selfFp32, dim, index, srcFp32);
    self.copy_(selfFp32);
  }

  if(self.scalar_type() != selfType){
    self = self.to(ScalarType::Half);
  }

  return self;
}

Tensor scatter_add_out_npu(
    Tensor& result,
    const Tensor& self,
    Dimname dim,
    const Tensor& index,
    const Tensor& src) {
  return scatter_add_out_npu(
      result, self, dimname_to_position(self, dim), index, src);
}

Tensor scatter_add_npu(
    const Tensor& self,
    Dimname dim,
    const Tensor& index,
    const Tensor& src) {
  return scatter_add_npu(self, dimname_to_position(self, dim), index, src);
}

} // namespace native
} // namespace at