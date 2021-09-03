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

#include "c10/npu/OptionsManager.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor&, Tensor&> topk_out_npu_no_transpose(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  SmallVector<int64_t, N> kVec = {k};
  Tensor kCpuTensor = from_blob((void*)kVec.data(), {1}, at::kLong).to(at::kInt);
  OpCommand cmd;
  cmd.Name("TopKV2")
    .Input(self)
    .Input(kCpuTensor, kVec, "k")
    .Output(values)
    .Output(indices)
    .Attr("dim", dim)
    .Attr("largest", largest)
    .Attr("sorted", sorted)
    .Run();
  
  return tuple<Tensor&, Tensor&>(values, indices);
}

tuple<Tensor&, Tensor&> topk_out_npu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::make_wrap_dim(-1, self.dim());

  if (dim != lastDim) {
    SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);

    // construct the output tensor of the NPU
    Tensor transposeSelf = at::npu_transpose(self, perm);
    auto outputSize = transpose_npu_output_size(values, perm);
    Tensor transposeValue = at::empty_with_format(
        outputSize,
        values.options(),
        CalcuOpUtil::get_tensor_npu_format(values));
    Tensor transposeIndices = at::empty_with_format(
        outputSize,
        indices.options(),
        CalcuOpUtil::get_tensor_npu_format(indices));
    topk_out_npu_no_transpose(
        transposeSelf,
        k,
        lastDim,
        largest,
        sorted,
        values,
        transposeIndices);
    at::npu_transpose_out(values, transposeValue, perm);
    at::npu_transpose_out(indices, transposeIndices, perm);
  } else {
    topk_out_npu_no_transpose(self, k, lastDim, largest, sorted, values, indices);
  }

  return tuple<Tensor&, Tensor&>(values, indices);
}

tuple<Tensor, Tensor> topk_npu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor selfCp = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);
  // construct the output tensor of the NPU
  Tensor values = at::empty_with_format(
      outputSize, selfCp.options(), CalcuOpUtil::get_tensor_npu_format(selfCp));
  Tensor indices = at::empty_with_format(
      outputSize, selfCp.options().dtype(kInt), ACL_FORMAT_ND);

  // calculate the output result of the NPU
  topk_out_npu(selfCp, k, dim, largest, sorted, values, indices);

  // indices dtype transform Int64
  indices = indices.to(at::kLong);

  return tuple<Tensor, Tensor>(values, indices);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("topk", TORCH_FN(topk_npu));
  m.impl("topk.values", TORCH_FN(topk_out_npu));
}

} // namespace native
} // namespace at