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
  int64_t index_dim = index.dim();
  auto index_sizes = index.sizes();
  auto self_dim = self.dim();
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "index.scalar_type() != ScalarType::Long");
  TORCH_CHECK(dim < index_dim, "dim must smaller than index.dim()");
  TORCH_CHECK(index_dim == self_dim, "index.dim() must eq to self.dim()");
  TORCH_CHECK(src.dim() == self_dim, "src.dim() must eq to self.dim()");

  Tensor src_flatten = src.reshape(-1);
  Tensor index_flatten = index.cpu().reshape(-1);
  std::vector<int64_t> index_sizes_new(index_sizes.begin(), index_sizes.end());
  index_sizes_new.push_back(index_dim);
  Tensor new_index = at::empty(index_sizes_new, index_flatten.options());
  new_index = new_index.reshape({-1, index_dim}).fill_(0);
  int64_t numel_num = index.numel();
  int64_t stride = 1;
  int64_t data_stride = index_dim;
  int64_t* org_data_ptr = index_flatten.data_ptr<int64_t>();
  int64_t* data_ptr = new_index.data_ptr<int64_t>();

  for (--index_dim; index_dim >= 0; index_dim--) {
    int64_t dim_size = index.size(index_dim);
    for (int64_t i = 0; i < numel_num; i++) {
      if (dim != index_dim) {
        if (i >= stride) {
          data_ptr[i * data_stride + index_dim] = (i / stride) % dim_size;
        }
      } else {
        data_ptr[i * data_stride + index_dim] = org_data_ptr[i];
      }
    }
    stride = stride * dim_size;
  }

  OpCommand cmd;
  cmd.Name("ScatterNdAdd")
     .Input(self)
     .Input(new_index.to("npu"))
     .Input(src_flatten)
     .Output(result)
     .Attr("use_locking", false)
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