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

namespace at_npu {
namespace native {

at::Tensor& argsort_out_npu_no_transpose(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
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

at::Tensor& argsort_out_npu_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  dim = make_wrap_dim(dim, self.dim());
  int64_t lastDim = make_wrap_dim(-1, self.dim());

  at::SmallVector<int64_t, SHAPE_SIZE> perm;
  for (int64_t i = 0; i < self.dim(); i++) {
    perm.emplace_back(i);
  }
  std::swap(perm[dim], perm[lastDim]);

  at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm);
  auto outputSize = transpose_npu_output_size(values, perm);
  at::Tensor transposeValues = OpPreparation::ApplyTensor(
      values,
      outputSize);
  at::Tensor transposeIndices = OpPreparation::ApplyTensor(
      indices,
      outputSize);

  argsort_out_npu_no_transpose(
      transposeValues, transposeIndices, transposeSelf, lastDim, descending);

  NPUNativeFunctions::npu_transpose_out(transposeIndices, perm, indices);
  
  // indices dtype transform to Int64
  indices = NPUNativeFunctions::npu_dtype_cast(indices, at::kLong);
  
  return indices;
}

at::Tensor NPUNativeFunctions::argsort(const at::Tensor& self,
    int64_t dim,
    bool descending) {
  // construct the output tensor of the NPU
  at::Tensor values = OpPreparation::ApplyTensor(self);
  at::Tensor indices = OpPreparation::ApplyTensor(self, self.options().dtype(at::kInt));
  // calculate the output result of the NPU
  argsort_out_npu_nocheck(values, indices, self, dim, descending);

  return indices;
}

at::Tensor NPUNativeFunctions::argsort(const at::Tensor& self,
    at::Dimname dim,
    bool descending) {
  return NPUNativeFunctions::argsort(self, dimname_to_position(self, dim), descending);
}

} // namespace native
} // namespace at_npu
