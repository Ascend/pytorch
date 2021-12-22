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

std::tuple<Tensor&, Tensor&> batch_norm_stats_out_npu_nocheck(
    Tensor& mean,
    Tensor& invstd,
    const Tensor& self,
    double eps) {
  SmallVector<int64_t, N> dim;
  int dimN = self.ndimension();
  for(int i = 0; i < dimN; i++){
    if (i == 1) {
      continue;
    }
    dim.emplace_back(i);
  }
  Tensor selfCp = self;
  if (self.scalar_type() != at::kFloat){
    selfCp = self.npu_dtype_cast(at::kFloat);
  }
  OpCommand cmd1;
  cmd1.Name("ReduceMean")
      .Input(selfCp)
      .Input(dim, at::kInt)
      .Output(mean)
      .Attr("keep_dims", (bool) false)
      .Run();

  Tensor meanCopy = mean;
  if (mean.dim() != 0) {
    auto dimVector = array_to_small_vector(dim);
    for (int64_t i = 0; i < dimVector.size(); i++) {
      meanCopy = meanCopy.unsqueeze(dimVector[i]);
    }
  }
  meanCopy = meanCopy.expand(self.sizes());
  OpCommand cmd2;
  cmd2.Name("ReduceStdWithMean")
      .Input(selfCp)
      .Input(meanCopy)
      .Output(invstd)
      .Attr("dim", dim)
      .Attr("unbiased", false)
      .Attr("keepdim", false)
      .Attr("invert", true)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();

  return std::tie(mean, invstd);
}

std::tuple<Tensor, Tensor> batch_norm_stats_npu(
    const Tensor& self,
    double eps) {
  TORCH_CHECK(
      self.ndimension() >= 2,
      "Expected 2D+ Tensor, but got tensor with ",
      self.ndimension(),
      " Dimension");
  Tensor mean = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  Tensor invstd = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  batch_norm_stats_out_npu_nocheck(mean, invstd, self, eps);

  return std::tie(mean, invstd);
}

} // namespace native
} // namespace at
