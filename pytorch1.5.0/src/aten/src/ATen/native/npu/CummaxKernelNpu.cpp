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

void cummax_out_npu_nocheck (   
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim) {
  OpCommand cmd;
  cmd.Name("Cummax")
    .Input(self)
    .Output(values)
    .Output(indices)
    .Attr("dim", dim)
    .Run();      
}

void cummax_helper_npu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  Tensor valuesTemp = OpPreparation::ApplyTensor(self);
  Tensor indicesTemp = OpPreparation::ApplyTensor(self, self.options().dtype(kLong));
    
  cummax_out_npu_nocheck(valuesTemp, indicesTemp, self, dim);

  values.copy_(valuesTemp);
  indices.copy_(indicesTemp);       
}

}}
