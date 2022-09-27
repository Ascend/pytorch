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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& repeat_interleave_out_npu(at::Tensor& result, at::Tensor& self, int64_t repeats, int64_t dim) {
  OpCommand cmd;
  cmd.Name("TileWithAxis")
    .Input(self)
    .Output(result)
    .Attr("tiles", repeats)
    .Attr("axis", dim)
    .Run();

  return result;
}

at::Tensor NPUNativeFunctions::repeat_interleave(
    const at::Tensor &self,
    int64_t repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t realDim = dim.value_or(0);

  int64_t self_dim = self.dim();
  if((realDim < -self_dim) || (realDim > self_dim - 1)){
    AT_ERROR("dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");
  }

  at::Tensor selfTensor = self;
  if(!dim.has_value()){
    selfTensor = at::flatten(selfTensor);
  }

  auto outputSize = repeat_interleave_npu_output_size(selfTensor, repeats, realDim);
  at::Tensor result = OpPreparation::ApplyTensor(selfTensor, outputSize);
  repeat_interleave_out_npu(result, selfTensor, repeats, realDim);

  return result;
}

} // namespace native
} // namespace at_npu