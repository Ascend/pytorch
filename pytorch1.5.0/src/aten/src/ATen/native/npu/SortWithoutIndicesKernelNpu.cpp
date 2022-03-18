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

Tensor& sort_without_indices_no_transpose(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    bool descending) {
  OpCommand cmd;
  cmd.Name("SortV2")
      .Input(self)
      .Output(result)
      .Attr("axis", dim)
      .Attr("descending", descending)
      .Run();
  
  return result;
}

Tensor& sort_without_indices_out_npu(
    Tensor& result,
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

    auto outputSize = transpose_npu_output_size(result, perm);

    Tensor transposeResult = at::empty_with_format(
        outputSize,
        result.options(),
        CalcuOpUtil::get_tensor_npu_format(result));

    sort_without_indices_no_transpose(
        transposeResult, transposeSelf, lastDim, descending);
      
    at::npu_transpose_out(result, transposeResult, perm);
  } else {
    sort_without_indices_no_transpose(
        result, self, dim, descending);
  }

  return result;
}

Tensor sort_without_indices_npu(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  auto outputSize = input_same_output_size(self);

  Tensor result = OpPreparation::ApplyTensor(self);
  
  sort_without_indices_out_npu(result, self, dim, descending);
  
  return result;
}
} // namespace native
} // namespace at