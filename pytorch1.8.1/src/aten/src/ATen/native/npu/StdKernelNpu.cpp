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

tuple<Tensor&, Tensor&> std_mean_out_npu_nocheck(
    Tensor& resultStd, 
    Tensor& resultMean, 
    const Tensor& self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  OpCommand cmd1;
  cmd1.Name("ReduceMeanD")
      .Input(self)
      .Output(resultMean)
      .Attr("axes", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  Tensor resultMeanCopy = resultMean;
  if (resultMean.dim() != 0 && keepdim == false) {
    auto dimVector = array_to_small_vector(dim);
    std::sort(dimVector.begin(), dimVector.end());
    for (int64_t i = 0; i < dimVector.size(); i++) {
      resultMeanCopy = resultMeanCopy.unsqueeze(dimVector[i]);
    }
  }
  resultMeanCopy = resultMeanCopy.expand(self.sizes());
  OpCommand cmd2;
  cmd2.Name("ReduceStdWithMean")
      .Input(self)
      .Input(resultMeanCopy)
      .Output(resultStd)
      .Attr("dim", dim)
      .Attr("unbiased", unbiased)
      .Attr("keepdim", keepdim)
      .Run();

  return std::tie(resultStd, resultMean);
}

Tensor& std_out_npu(
    const Tensor& self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim,
    Tensor& result) {
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  Tensor meanResult = OpPreparation::ApplyTensor(self, outputSize);

  OpPreparation::CheckOut(
      {self}, 
      result, 
      ACL_FORMAT_ND,
      self.scalar_type(),
      outputSize);

  std_mean_out_npu_nocheck(result, meanResult, self, dim, unbiased, keepdim);

  return result;
}

Tensor& std_names_out_npu(
    const Tensor& self, 
    DimnameList dim, 
    bool unbiased, 
    bool keepdim,
    Tensor& result) {
  return std_out_npu(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

Tensor std_dim_npu(
    const Tensor & self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

  Tensor result1 = OpPreparation::ApplyTensor(self, outputSize);
  Tensor result2 = OpPreparation::ApplyTensor(self, outputSize);

  std_mean_out_npu_nocheck(result1, result2, self, dim, unbiased, keepdim);
  return result1;
}

Tensor std_npu(
    const Tensor & self, 
    bool unbiased) {
  SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return std_dim_npu(self, dims, unbiased, false);
}

tuple <Tensor, Tensor> std_mean_dim_npu(
    const Tensor & self, 
    IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

  Tensor result1 = OpPreparation::ApplyTensor(self, outputSize);
  Tensor result2 = OpPreparation::ApplyTensor(self, outputSize);

  std_mean_out_npu_nocheck(result1, result2, self, dim, unbiased, keepdim);

  return std::tie(result1, result2);
}

tuple <Tensor, Tensor> std_mean_npu(
    const Tensor & self, 
    bool unbiased) {
  SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return std_mean_dim_npu(self, dims, unbiased, false);
}

tuple <Tensor, Tensor> std_mean_names_dim_npu(
    const Tensor & self, 
    DimnameList dim, 
    bool unbiased, 
    bool keepdim) {
  return std_mean_dim_npu(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor std_names_dim_npu(
    const Tensor & self, 
    DimnameList dim, 
    bool unbiased, 
    bool keepdim) {
  return std_dim_npu(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("std", TORCH_FN(std_npu));
  m.impl("std.dim", TORCH_FN(std_dim_npu));
  m.impl("std_mean", TORCH_FN(std_mean_npu));
  m.impl("std_mean.dim", TORCH_FN(std_mean_dim_npu));
  m.impl("std_mean.names_dim", TORCH_FN(std_mean_names_dim_npu));
  m.impl("std.out", TORCH_FN(std_out_npu));
  m.impl("std.names_dim", TORCH_FN(std_names_dim_npu));
  m.impl("std.names_out", TORCH_FN(std_names_out_npu));
}
} // namespace native
} // namespace at::native
