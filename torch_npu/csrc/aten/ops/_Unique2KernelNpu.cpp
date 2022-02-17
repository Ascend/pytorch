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

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> _unique2_out_npu(
    at::Tensor& y,
    at::Tensor& yOutputSize,
    at::Tensor& yInverse,
    at::Tensor& yCounts,
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  OpCommand cmd;
  cmd.Name("UniqueWithCountsAndSorting")
     .Input(self)
     .Output(y)
     .Output(yOutputSize)
     .Output(yInverse)
     .Output(yCounts)
     .Attr("sorted", sorted)
     .Attr("return_inverse", true)
     .Attr("return_counts", true)
     .Run();

  return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>(y, yOutputSize, yInverse, yCounts);
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::_unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  if (self.numel() == 0) {
    at::Tensor result= OpPreparation::ApplyTensor(self, {0});
    at::Tensor yInverse = OpPreparation::ApplyTensor({0}, self.options().dtype(at::kLong), self);
    at::Tensor yCounts = OpPreparation::ApplyTensor({0}, self.options().dtype(at::kLong), self);
    return std::tie(result, yInverse, yCounts);
  }
  
  auto yInverseSize = input_same_output_size(self);
  auto outputSizes = tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>, at::IntArrayRef>(
      {self.numel()}, {1}, yInverseSize);

  at::Tensor selfCopy = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfCopy = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
 
  at::Tensor y = OpPreparation::ApplyTensor(selfCopy, std::get<0>(outputSizes));
  at::Tensor yOutputSize = OpPreparation::ApplyTensorWithSizes(std::get<1>(outputSizes), self.options().dtype(at::kLong));
  at::Tensor yInverse = OpPreparation::ApplyTensorWithSizes(std::get<2>(outputSizes), self.options().dtype(at::kLong));
  at::Tensor yCounts = OpPreparation::ApplyTensorWithSizes(std::get<0>(outputSizes), self.options().dtype(at::kLong));
  
  _unique2_out_npu(y, yOutputSize, yInverse, yCounts, selfCopy, sorted, return_inverse, return_counts);
  
  int64_t count = yOutputSize[0].item().toLong();
  at::Tensor result = y.slice(0, 0, count, 1);
  result = NpuUtils::format_contiguous(result);

  if (self.scalar_type() == at::ScalarType::Half) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }

  if (return_counts) {
    yCounts = yCounts.slice(0, 0, count, 1);
    yCounts = NpuUtils::format_contiguous(yCounts);
  } else {
    yCounts = at::empty({0}, self.options().dtype(at::kLong));
  }
  
  if (!(return_counts || return_inverse)) {
    yInverse = at::empty({0}, self.options().dtype(at::kLong));
  }
  
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(result, yInverse, yCounts);
}
} // namespace native
} // namespace at_npu