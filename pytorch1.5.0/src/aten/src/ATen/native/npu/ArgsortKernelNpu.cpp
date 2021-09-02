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

Tensor& argsort_out_npu_no_transpose(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool descending) {
  OpCommand cmd;
  cmd.Name("Sort")
     .Input(self)
     .Output(values)
     .Output(indices)
     .Attr("axis", dim)
     .Attr("descending", descending)
     .Run();

  return indices;
}

Tensor& argsort_out_npu_nocheck(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool descending) {
  dim = make_wrap_dim(dim, self.dim());
  int64_t lastDim = make_wrap_dim(-1, self.dim());

  SmallVector<int64_t, SHAPE_SIZE> perm;
  for (int64_t i = 0; i < self.dim(); i++) {
    perm.emplace_back(i);
  }
  std::swap(perm[dim], perm[lastDim]);

  Tensor transposeSelf = at::npu_transpose(self, perm);
  auto outputSize = transpose_npu_output_size(values, perm);
  Tensor transposeValues = OpPreparation::ApplyTensor(
      values,
      outputSize);
  Tensor transposeIndices = OpPreparation::ApplyTensor(
      indices,
      outputSize);

  argsort_out_npu_no_transpose(
      transposeValues, transposeIndices, transposeSelf, lastDim, descending);

  at::npu_transpose_out(indices, transposeIndices, perm);
  
  //indices dtype transform to Int64
  indices = indices.to(at::kLong);
  
  return indices;
}

Tensor argsort_npu(const Tensor& self,
    int64_t dim,
    bool descending) {
  // construct the output tensor of the NPU
  Tensor values = OpPreparation::ApplyTensor(self);
  Tensor indices = OpPreparation::ApplyTensor(self, self.options().dtype(kInt));
  // calculate the output result of the NPU
  argsort_out_npu_nocheck(values, indices, self, dim, descending);

  return indices;
}

Tensor argsort_npu(const Tensor& self,
    Dimname dim,
    bool descending) {
  return argsort_npu(self, dimname_to_position(self, dim), descending);
}

} // namespace native
} // namespace at
