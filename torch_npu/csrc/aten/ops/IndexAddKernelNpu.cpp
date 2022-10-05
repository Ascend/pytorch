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
#include<ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor& index_add_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor indices = index;
  if (index.scalar_type() != at::ScalarType::Int) {
    indices = index.to(at::kInt);
  }
  if (index.dim() == 0) {
    indices.unsqueeze_(0);
  }
  
  at::SmallVector<int64_t, N> pad_size = array_to_small_vector(self.sizes());
  pad_size[dim] = indices.sizes()[0];
  at::Tensor source_broadcast = NPUNativeFunctions::npu_broadcast(source, pad_size);
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
 
at::Tensor& NPUNativeFunctions::index_add_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  OpPreparation::CheckMemory({self, index, source}, {self});
  if (!NpuUtils::check_match(&self)) {
      at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      at::Tensor result = index_add_out_npu(contiguousSelf, contiguousSelf, dim, index, source);
      NpuUtils::format_fresh_view(self, result);
  } else {
      index_add_out_npu(self, self, dim, index, source);
  }
  return self;
}

at::Tensor NPUNativeFunctions::index_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  return self.clone().index_add_(dim, index, source);
}

at::Tensor NPUNativeFunctions::index_add(
    const at::Tensor& self,
    at::Dimname dim, 
    const at::Tensor& index,
    const at::Tensor& source)  {
  return NPUNativeFunctions::index_add(self, dimname_to_position(self, dim), index, source);
}
} // namespace native
} // namespace at_npu