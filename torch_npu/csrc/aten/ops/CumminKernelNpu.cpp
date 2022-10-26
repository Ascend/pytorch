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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
void cummin_out_npu_nocheck (   
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim) {
  OpCommand cmd;
  cmd.Name("Cummin")
    .Input(self)
    .Output(values)
    .Output(indices)
    .Attr("axis", dim)
    .Run();      
}

void NPUNativeFunctions::_cummin_helper(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim) {   
  // process aicpu
  if(self.scalar_type() == at::ScalarType::Long){
    at::Tensor valuesTemp = OpPreparation::ApplyTensor(self);
    at::Tensor indicesTemp = OpPreparation::ApplyTensor(self, self.options().dtype(at::kLong)); 
    cummin_out_npu_nocheck(valuesTemp, indicesTemp, self, dim);
    values.copy_(valuesTemp);
    indices.copy_(indicesTemp);
  } else {
    // process aicore
    int64_t firstDim = CalcuOpUtil::make_wrap_dim(0, self.dim());
    if (dim != firstDim) {
      c10::SmallVector<int64_t, SHAPE_SIZE> perm;
      for (int64_t i = 0; i < self.dim(); i++) {
        perm.emplace_back(i);
      }
      std::swap(perm[dim], perm[firstDim]);
      
      at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm, true);      
      auto outputSize = transpose_npu_output_size(values, perm);      
      at::Tensor transposeValue = OpPreparation::ApplyTensor(self, outputSize);
      at::Tensor transposeIndices = OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::kInt), self); 

      cummin_out_npu_nocheck(transposeValue, transposeIndices, transposeSelf, firstDim);
      // Indices must to be long
      transposeIndices = NPUNativeFunctions::npu_dtype_cast(transposeIndices, at::kLong);
      NPUNativeFunctions::npu_transpose_out(transposeValue, perm, true, values);
      NPUNativeFunctions::npu_transpose_out(transposeIndices, perm, true, indices);
    } else {
      at::Tensor valuesTemp = OpPreparation::ApplyTensor(self);
      at::Tensor indicesTemp = OpPreparation::ApplyTensor(self, self.options().dtype(at::kInt)); 
      cummin_out_npu_nocheck(valuesTemp, indicesTemp, self, dim);
      indicesTemp = NPUNativeFunctions::npu_dtype_cast(indicesTemp, at::kLong);
      values.copy_(valuesTemp);
      indices.copy_(indicesTemp);
    }  
  }     
}

}}
