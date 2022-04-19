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

Tensor& repeat_interleave_out_npu(Tensor& result, Tensor& self, int64_t repeats, int64_t dim) {
  OpCommand cmd;
  cmd.Name("TileWithAxis")
      .Input(self)
      .Output(result)
      .Attr("tiles", repeats)
      .Attr("axis", dim)
      .Run();
  return result;
}

Tensor repeat_interleave_npu(const Tensor &self, int64_t repeats, c10::optional<int64_t> dim) {
  int64_t realDim = dim.value_or(0);
  
  // dim value must be greater than or equal to 0.
  int64_t self_dim = self.dim();
  if((realDim < -self_dim) || (realDim > self_dim - 1)){
    AT_ERROR("dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");
  }

  Tensor selfTensor = self;
  if(!dim.has_value()){
    selfTensor = at::flatten(selfTensor);
  }
  // calculate the output size
  auto outputSize = repeat_interleave_npu_output_size(selfTensor, repeats, realDim);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(
      selfTensor, outputSize);

  // calculate the output result of the NPU
  repeat_interleave_out_npu(result, selfTensor, repeats, realDim);

  return result;
}

} // namespace native
} // namespace at