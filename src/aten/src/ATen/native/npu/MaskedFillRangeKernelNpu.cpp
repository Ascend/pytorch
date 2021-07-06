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

void mask_fill_range_check(
    const Tensor& self,
    const Tensor& start,
    const Tensor& end,
    const Tensor& value,
    int64_t axis){
  int64_t x_dim = self.dim();
  int64_t min = -x_dim;
  int64_t max = x_dim - 1;
  TORCH_CHECK(
      !(axis < min || axis > max),
      "axis overfloaw the range, expected in range [",
      -x_dim,
      " ",
      x_dim - 1,
      "] ");
  TORCH_CHECK(
      start.ndimension() == 2 && start.sizes() == end.sizes(),
      "Expected noempty 2D start tensor and start' sizes() should be equal end's sizes() ");
  TORCH_CHECK(
      start.size(0) == value.size(0),
      "Expected value.length equal start loop num ");
  TORCH_CHECK(
      self.scalar_type() == value.scalar_type(),
      "value dtype should be equal self dtype !, but value dtype is ",
      value.scalar_type(),
      " and self dtype is ",
      self.scalar_type());
}

Tensor masked_fill_range_npu(
    const Tensor& self,
    const Tensor& start,
    const Tensor& end,
    const Tensor& value,
    int64_t axis){
  mask_fill_range_check(self, start, end, value, axis);
  Tensor result = OpPreparation::ApplyTensor(self);
  OpCommand cmd;
  cmd.Name("MaskedFillRange")
      .Input(self)
      .Input(start)
      .Input(end)
      .Input(value)
      .Output(result)
      .Attr("axis", axis)
      .Run();
  return result;
}

}
}
