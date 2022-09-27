// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

std::tuple<Tensor&, Tensor&, Tensor&> _unique2_out_npu(
    Tensor& y,
    Tensor& yInverse,
    Tensor& yCounts,
    const Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  SmallVector<int64_t, N> output_sync_idx = {0, 2};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("UniqueWithCountsAndSorting")
      .Input(self)
      .Output(y)
      .Output(yInverse)
      .Output(yCounts)
      .Attr("sorted", sorted)
      .Attr("return_inverse", return_inverse)
      .Attr("return_counts", return_counts)
      .Run();

  return std::tuple<Tensor&, Tensor&, Tensor&>(y, yInverse, yCounts);
}

tuple<Tensor, Tensor, Tensor> _unique2_npu(
    const Tensor& selfOp,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  // Data accuracy loss in fp16 scene
  const Tensor self = selfOp.scalar_type() == at::kHalf ? selfOp.npu_dtype_cast(at::kFloat) : selfOp;
  if (self.numel() == 0) {
    Tensor result= OpPreparation::ApplyTensor(self, {0});
    Tensor yInverse = OpPreparation::ApplyTensor({0}, self.options().dtype(kLong), self);
    Tensor yCounts = OpPreparation::ApplyTensor({0}, self.options().dtype(kLong), self);
    return std::tie(result, yInverse, yCounts);
  }

  Tensor y = OpPreparation::ApplyTensor(self, self.numel());
  Tensor yInverse = !(return_counts || return_inverse) ?
      OpPreparation::ApplyTensorWithFormat({0}, self.options().dtype(kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options().dtype(kLong), ACL_FORMAT_ND);
  Tensor yCounts = return_counts ?
      OpPreparation::ApplyTensorWithFormat(self.numel(), self.options().dtype(kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat({0}, self.options().dtype(kLong), ACL_FORMAT_ND);
  
  _unique2_out_npu(y, yInverse, yCounts, self, sorted, return_inverse, return_counts);
  if (selfOp.scalar_type() == at::kHalf) {
    y = y.npu_dtype_cast(at::kHalf);
  }

  return std::tuple<Tensor, Tensor, Tensor>(y, yInverse, yCounts);
}

} // namespace native
} // namespace at
