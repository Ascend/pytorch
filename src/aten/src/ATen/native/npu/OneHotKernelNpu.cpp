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

Tensor one_hot_npu1(const Tensor& self, int64_t num_classes) {
  Scalar on_value = 1;
  Scalar off_value = 0;
  int64_t axis = -1;
  int64_t depth;

  auto self_temp = self.to(at::kFloat);

  //When run in NPU,the input tensor's dim must be smaller than 8.
  TORCH_CHECK(
      self_temp.dim() < 8, "NPU error,can not support the input tensor's dim bigger than 7.");

  // empty tensor could be converted to one hot representation,
  // but shape inference is not possible.
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      depth = num_classes;
    }
  }

  // non-empty tensor
  TORCH_CHECK(
      self_temp.min().item().toLong() >= 0, "Class values must be non-negative.");
  if (num_classes == -1) {
    depth = self_temp.max().item().toLong() + 1;
  } else {
    TORCH_CHECK(
        num_classes > self_temp.max().item().toLong(),
        "Class values must be smaller than num_classes.");
    depth = num_classes;
  }

  // calculate output size
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.emplace_back(depth);

  Tensor result = OpPreparation::ApplyTensor(
      outputSize,
      self.options().dtype(ScalarType::Int),
      self);

  SmallVector<int64_t, N> depthList = {depth};
  
  OpCommand cmd;
  cmd.Name("OneHot")
      .Input(self)
      .Input(depthList, at::kInt)
      .Input(on_value, ScalarType::Int)
      .Input(off_value, ScalarType::Int)
      .Output(result)
      .Attr("axis", axis)
      .Run();
  
  return result;
}

} // namespace native
} // namespace at
