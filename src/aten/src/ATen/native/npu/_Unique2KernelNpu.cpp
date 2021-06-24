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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _unique2_out_npu(
    Tensor& y,
    Tensor& yOutputSize,
    Tensor& yInverse,
    Tensor& yCounts,
    const Tensor& self,
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

  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(y, yOutputSize, yInverse, yCounts);
}

tuple<Tensor, Tensor, Tensor> _unique2_npu(
    const Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  if(self.numel() == 0){
    Tensor result= at::empty_with_format({0}, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
    Tensor yInverse = at::empty_with_format({0}, self.options().dtype(kLong), CalcuOpUtil::get_tensor_npu_format(self));
    Tensor yCounts = at::empty_with_format({0}, self.options().dtype(kLong), CalcuOpUtil::get_tensor_npu_format(self));
    return std::tie(result, yInverse, yCounts);
  }
  
  auto yInverseSize = input_same_output_size(self);
  auto outputSizes = tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>, IntArrayRef>(
    {self.numel()}, {1}, yInverseSize);

  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = self.to(ScalarType::Float);
  }
 
  Tensor y = at::empty_with_format(std::get<0>(outputSizes), selfCopy.options(), CalcuOpUtil::get_tensor_npu_format(selfCopy));
  Tensor yOutputSize = at::empty_with_format(std::get<1>(outputSizes), self.options().dtype(kLong), ACL_FORMAT_ND);
  Tensor yInverse = at::empty_with_format(std::get<2>(outputSizes), self.options().dtype(kLong), ACL_FORMAT_ND);
  Tensor yCounts = at::empty_with_format(std::get<0>(outputSizes), self.options().dtype(kLong), ACL_FORMAT_ND);
  
  _unique2_out_npu(y, yOutputSize, yInverse, yCounts, selfCopy, sorted, return_inverse, return_counts);
  
  int64_t count = yOutputSize[0].item().toLong();
  Tensor result = y.slice(0, 0, count, 1);
  result = NpuUtils::format_contiguous(result);

  if (self.scalar_type() == ScalarType::Half) {
    result = result.to(ScalarType::Half);
  }

  if (return_counts) {
    yCounts = yCounts.slice(0, 0, count, 1);
    yCounts = NpuUtils::format_contiguous(yCounts);
  } else {
    yCounts = at::empty({0}, self.options().dtype(kLong));
  }
  
  if (!(return_counts || return_inverse)) {
    yInverse = at::empty({0}, self.options().dtype(kLong));
  }
  
  return std::tuple<Tensor, Tensor, Tensor>(result, yInverse, yCounts);
}

} // namespace native
} // namespace at
