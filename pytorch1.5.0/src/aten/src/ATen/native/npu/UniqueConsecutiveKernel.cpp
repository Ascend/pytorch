// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

std::tuple<Tensor&, Tensor&, Tensor&> unique_consecutive_npu_nocheck(
    Tensor& output,
    Tensor& inverse_indices,
    Tensor& counts,
    const Tensor& self, 
    const bool return_inverse, 
    const bool return_counts, 
    c10::optional<int64_t> dim) {
  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = at::npu_dtype_cast(self, ScalarType::Float);
    output = at::npu_dtype_cast(output, ScalarType::Float);
  }
  SmallVector<int64_t, N> output_sync_idx = {0, 2};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
     .Name("UniqueConsecutive")
     .Input(selfCopy)
     .Output(output)
     .Output(inverse_indices)
     .Output(counts)
     .Attr("return_idx", return_inverse)
     .Attr("return_counts", return_counts);
  if (dim.has_value()) {
    cmd.Attr("axis", dim.value());
  }
  cmd.Run();
  if (self.scalar_type() == ScalarType::Half) {
    output = at::npu_dtype_cast(output, ScalarType::Half);
  }
  return std::tie(output, inverse_indices, counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_npu(
    const Tensor& self, 
    const bool return_inverse, 
    const bool return_counts, 
    c10::optional<int64_t> dim) {
  Tensor output = dim.has_value() ? 
      OpPreparation::ApplyTensor(self) : OpPreparation::ApplyTensor(self, {self.numel()});
  Tensor inverse_indices = dim.has_value() ? 
      OpPreparation::ApplyTensorWithFormat(self.size(dim.value()), self.options().dtype(kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options().dtype(kLong), ACL_FORMAT_ND);
  Tensor counts = dim.has_value() ? 
      OpPreparation::ApplyTensorWithFormat(self.size(dim.value()), self.options().dtype(kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat({self.numel()}, self.options().dtype(kLong), ACL_FORMAT_ND);
  unique_consecutive_npu_nocheck(output, inverse_indices, counts, self, return_inverse, return_counts, dim);
  return std::tie(output, inverse_indices, counts);
}


} // namespace native
} // namespace at