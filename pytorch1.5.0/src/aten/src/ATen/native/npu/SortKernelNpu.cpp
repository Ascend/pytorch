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

tuple<Tensor&, Tensor&> sort_out_npu_no_transpose(
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

  return std::tie(values, indices);
}

tuple<Tensor&, Tensor&> sort_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool descending) {
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::make_wrap_dim(-1, self.dim());

  if (dim != lastDim) {
    SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);

    Tensor transposeSelf = at::npu_transpose(self, perm);
    auto outputSize = transpose_npu_output_size(values, perm);
    Tensor transposeValues = at::empty_with_format(
        outputSize,
        values.options(),
        CalcuOpUtil::get_tensor_npu_format(values));
    Tensor transposeIndices = at::empty_with_format(
        outputSize,
        indices.options(),
        CalcuOpUtil::get_tensor_npu_format(indices));

    sort_out_npu_no_transpose(
      transposeValues, transposeIndices, transposeSelf, lastDim, descending);
    
    at::npu_transpose_out(values, transposeValues, perm);
    at::npu_transpose_out(indices, transposeIndices, perm);
  } else {
    sort_out_npu_no_transpose(
        values, indices, self, lastDim, descending);
  }
  
  // indices dtype transform Int64
  indices = indices.to(at::kLong);
  
  return std::tie(values, indices);
}

tuple<Tensor&, Tensor&> sort_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Dimname dim,
    bool descending) {
  return sort_out_npu(
      values, indices, self, dimname_to_position(self, dim), descending);
}

tuple<Tensor, Tensor> sort_npu(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  auto outputSize = input_same_output_size(self);

  Tensor values = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  Tensor indices = at::empty_with_format(
      outputSize, self.options().dtype(kInt), ACL_FORMAT_NCHW);

  sort_out_npu(values, indices, self, dim, descending);

  return std::tie(values, indices);
}

tuple<Tensor, Tensor> sort_npu(
    const Tensor& self,
    Dimname dim,
    bool descending) {
  return sort_npu(self, dimname_to_position(self, dim), descending);
}

} // namespace native
} // namespace at