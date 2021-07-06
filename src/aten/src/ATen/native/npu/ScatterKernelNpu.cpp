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

Tensor& scatter_npu_(
    Tensor& self,
    int64_t dim,
    const Tensor& index_ex,
    const Tensor& src_ex) {
  ScalarType selfType = self.scalar_type();

  if (self.scalar_type() == ScalarType::Half) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }

  Tensor index = index_ex; 
  if (index.scalar_type() == ScalarType::Half) {
    index = index.npu_dtype_cast(ScalarType::Float);
  }

  Tensor src = src_ex;
  if (src.scalar_type() == ScalarType::Half) {
    src = src.npu_dtype_cast(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("ScatterTensor")
     .Input(index)
     .Input(src)
     .Output(self)
     .Attr("dim", dim)
     .Run();

  if(self.scalar_type() != selfType){
    self = self.to(ScalarType::Half);
  }
  
  return self;
}

Tensor& scatter_npu_(
    Tensor& self,
    int64_t dim,
    const Tensor& index_ex,
    Scalar src) {
  ScalarType selfType = self.scalar_type();

  if (self.scalar_type() == ScalarType::Half) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }

  Tensor index = index_ex;
  if (index.scalar_type() == ScalarType::Half) {
    index = index.npu_dtype_cast(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("ScatterScalar")
     .Input(index)
     .Output(self)
     .Attr("dim", dim)
     .Attr("value", src)
     .Run();

  if(self.scalar_type() != selfType){
    self = self.to(ScalarType::Half);
  }

  return self;
}

} // namespace native
} // namespace at