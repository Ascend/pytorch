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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> _unique2_out_npu(
    at::Tensor& y,
    at::Tensor& yInverse,
    at::Tensor& yCounts,
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  c10::SmallVector<int64_t, N> output_sync_idx = {0, 2};
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

  return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(y, yInverse, yCounts);
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::_unique2(
    const at::Tensor& selfOp,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  // Data accuracy loss in fp16 scene
  const at::Tensor self = selfOp.scalar_type() == at::kHalf ? NPUNativeFunctions::npu_dtype_cast(selfOp, at::kFloat) : selfOp;
  
  if (self.numel() == 0) {
    at::Tensor result = OpPreparation::ApplyTensor(self, {0});
    at::Tensor yInverse = OpPreparation::ApplyTensor({0}, self.options().dtype(at::kLong), self);
    at::Tensor yCounts = OpPreparation::ApplyTensor({0}, self.options().dtype(at::kLong), self);
    return std::tie(result, yInverse, yCounts);
  }
  at::Tensor y = OpPreparation::ApplyTensor(self, self.numel());
  at::Tensor yInverse = !(return_counts || return_inverse) ?
      OpPreparation::ApplyTensorWithFormat({0}, self.options().dtype(at::kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options().dtype(at::kLong), ACL_FORMAT_ND);
  at::Tensor yCounts = return_counts ?
      OpPreparation::ApplyTensorWithFormat(self.numel(), self.options().dtype(at::kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat({0}, self.options().dtype(at::kLong), ACL_FORMAT_ND);

  _unique2_out_npu(y, yInverse, yCounts, self, sorted, return_inverse, return_counts);
  if (selfOp.scalar_type() == at::kHalf) {
    y = NPUNativeFunctions::npu_dtype_cast(y, at::kHalf);
  }

  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, yInverse, yCounts);
}
} // namespace native
} // namespace at_npu
