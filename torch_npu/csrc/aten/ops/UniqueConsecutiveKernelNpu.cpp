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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> unique_consecutive_out_npu(
    at::Tensor& output,
    at::Tensor& inverse_indices,
    at::Tensor& counts,
    const at::Tensor& self, 
    const bool return_inverse, 
    const bool return_counts, 
    c10::optional<int64_t> dim) {
  at::Tensor selfCopy = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfCopy = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    output = NPUNativeFunctions::npu_dtype_cast(output, at::ScalarType::Float);
  }
  c10::SmallVector<int64_t, N> output_sync_idx = {0, 2};
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
  if (self.scalar_type() == at::ScalarType::Half) {
    output = NPUNativeFunctions::npu_dtype_cast(output, at::ScalarType::Half);
  }
  return std::tie(output, inverse_indices, counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::unique_consecutive(
    const at::Tensor& self, 
    const bool return_inverse, 
    const bool return_counts, 
    c10::optional<int64_t> dim) {
  at::Tensor output = (dim.has_value()) ? 
      OpPreparation::ApplyTensor(self) : OpPreparation::ApplyTensor(self, {self.numel()});
  at::Tensor inverse_indices = (dim.has_value()) ? 
      OpPreparation::ApplyTensorWithFormat(self.size(dim.value()), self.options().dtype(at::kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options().dtype(at::kLong), ACL_FORMAT_ND);
  at::Tensor counts = (dim.has_value()) ? 
      OpPreparation::ApplyTensorWithFormat(self.size(dim.value()), self.options().dtype(at::kLong), ACL_FORMAT_ND) :
      OpPreparation::ApplyTensorWithFormat({self.numel()}, self.options().dtype(at::kLong), ACL_FORMAT_ND);
  unique_consecutive_out_npu(output, inverse_indices, counts, self, return_inverse, return_counts, dim);
  return std::tie(output, inverse_indices, counts);
}


} // namespace native
} // namespace at