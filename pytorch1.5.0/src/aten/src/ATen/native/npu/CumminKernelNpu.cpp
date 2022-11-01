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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

void cummin_out_npu_nocheck (   
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim) {
  OpCommand cmd;
  cmd.Name("Cummin")
    .Input(self)
    .Output(values)
    .Output(indices)
    .Attr("axis", dim)
    .Run();      
}

void cummin_helper_npu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {   
  // process aicpu
  if(self.scalar_type() == ScalarType::Long){
    Tensor valuesTemp = OpPreparation::ApplyTensor(self);
    Tensor indicesTemp = OpPreparation::ApplyTensor(self, self.options().dtype(kLong)); 
    cummin_out_npu_nocheck(valuesTemp, indicesTemp, self, dim);
    values.copy_(valuesTemp);
    indices.copy_(indicesTemp);
  } else {
    // process aicore
    int64_t firstDim = CalcuOpUtil::make_wrap_dim(0, self.dim());
    if (dim != firstDim) {
      SmallVector<int64_t, SHAPE_SIZE> perm;
      for (int64_t i = 0; i < self.dim(); i++) {
        perm.emplace_back(i);
      }
      std::swap(perm[dim], perm[firstDim]);
      
      Tensor transposeSelf = at::npu_transpose(self, perm);      
      auto outputSize = transpose_npu_output_size(values, perm);      
      Tensor transposeValue = OpPreparation::ApplyTensor(self, outputSize);
      Tensor transposeIndices = OpPreparation::ApplyTensor(outputSize, self.options().dtype(kInt), self); 

      cummin_out_npu_nocheck(transposeValue, transposeIndices, transposeSelf, firstDim);
      // Indices must to be long
      transposeIndices = transposeIndices.npu_dtype_cast(kLong);
      at::npu_transpose_out(values, transposeValue, perm);
      at::npu_transpose_out(indices, transposeIndices, perm);
    } else {
      Tensor valuesTemp = OpPreparation::ApplyTensor(self);
      Tensor indicesTemp = OpPreparation::ApplyTensor(self, self.options().dtype(kInt)); 
      cummin_out_npu_nocheck(valuesTemp, indicesTemp, self, dim);
      indicesTemp = indicesTemp.npu_dtype_cast(kLong);
      values.copy_(valuesTemp);
      indices.copy_(indicesTemp);
    }  
  }     
}

}}
