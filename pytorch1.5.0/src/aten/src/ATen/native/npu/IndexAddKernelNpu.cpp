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
#include<ATen/NamedTensorUtils.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& index_add_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  Tensor indices = index;
  if (index.scalar_type() != at::ScalarType::Int) {
    indices = index.npu_dtype_cast(at::kInt);
  }
  if (index.dim() == 0) {
    indices.unsqueeze_(0);
  }

  SmallVector<int64_t, N> pad_size = array_to_small_vector(self.sizes());
  pad_size[dim] = indices.sizes()[0];
  Tensor source_broadcast = at::npu_broadcast(source, pad_size);
  OpCommand cmd;
  cmd.Name("InplaceIndexAdd")
      .Input(self)
      .Input(indices)
      .Input(source_broadcast)
      .Output(result)
      .Attr("axis", dim)
      .Run();
  return result;
}

Tensor& index_add_npu_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  OpPreparation::CheckMemory({self, index, source}, {self});
  if (!NpuUtils::check_match(&self)) {
      Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      Tensor result = index_add_out_npu(contiguousSelf, contiguousSelf, dim, index, source);
      NpuUtils::format_fresh_view(self, result);
  } else {
      index_add_out_npu(self, self, dim, index, source);
  }
  return self;
}

Tensor index_add_npu(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  return self.clone().index_add_(dim, index, source);
}

Tensor index_add_npu(
    const Tensor& self,
    Dimname dim, 
    const Tensor& index,
    const Tensor& source)  {
  return index_add_npu(self, dimname_to_position(self, dim), index, source);
}

} // namespace native
} // namespace at
