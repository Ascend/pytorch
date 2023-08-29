// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

std::tuple<at::Tensor&, at::Tensor&> _unique_out_npu(
    at::Tensor& y,
    at::Tensor& yInverse,
    const at::Tensor& self,
    bool sorted,
    bool return_inverse) {
  c10::SmallVector<int64_t, N> output_sync_idx = {0, 1};
  at::Tensor yCounts = OpPreparation::ApplyTensorWithFormat({1}, self.options().dtype(at::kLong), ACL_FORMAT_ND);
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("UniqueWithCountsAndSorting")
      .Input(self)
      .Output(y)
      .Output(yInverse)
      .Output(yCounts)
      .Attr("sorted", sorted)
      .Attr("return_inverse", return_inverse)
      .Attr("return_counts", false)
      .Run();

  return std::tuple<at::Tensor&, at::Tensor&>(y, yInverse);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_unique(
    const at::Tensor& selfOp,
    bool sorted,
    bool return_inverse) {
  /*
   * The std::unordered_set called by the operator deduplication will shuffle the order according to the hash function, 
   * and the fp16 scene is shuffled differently from the basic data type, so that when sorted=false, the fp16 accuracy 
   * is not up to standard. In addition, when the operator is deduplicated, FP16 has data accuracy loss, so FP16 is 
   * strongly converted to FP32 here.
   */
  const at::Tensor self = selfOp.scalar_type() == at::kHalf ? NPUNativeFunctions::npu_dtype_cast(selfOp, at::kFloat) : selfOp;
  
  if (self.numel() == 0) {
    at::Tensor result = OpPreparation::ApplyTensor(self, {0});
    at::Tensor yInverse = OpPreparation::ApplyTensor({0}, self.options().dtype(at::kLong), self);
    return std::tie(result, yInverse);
  }
  at::Tensor y = OpPreparation::ApplyTensor(self, self.numel());
  at::Tensor yInverse = !return_inverse ?
      OpPreparation::ApplyTensorWithFormat({1}, self.options().dtype(at::kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options().dtype(at::kLong), ACL_FORMAT_ND);

  _unique_out_npu(y, yInverse, self, sorted, return_inverse);
  if (selfOp.scalar_type() == at::kHalf) {
    y = NPUNativeFunctions::npu_dtype_cast(y, at::kHalf);
  }

  return std::tuple<at::Tensor, at::Tensor>(y, yInverse);
}
} // namespace native
} // namespace at_npu
