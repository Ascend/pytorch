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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& NPUNativeFunctions::adaptive_avg_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& result) {
  if (output_size[0] == 1 && output_size[1] == 1) {
    at::mean_out(result, self, {self.dim() - 2, self.dim() - 1}, true);
  } else {
    OpCommand cmd;
    cmd.Name("AdaptiveAvgPool2d")
        .Input(self)
        .Output(result)
        .Attr("output_size", output_size)
        .Run();
  }

  return result;
}

at::Tensor NPUNativeFunctions::adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
  // The logic is a little different from CPU_impl
  return at::_adaptive_avg_pool2d(self, output_size);
}

at::Tensor NPUNativeFunctions::_adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
  for (int64_t i = 0; i < self.dim(); i++) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pooling2d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (self.dim() == 3 || self.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  auto outputSize = array_to_small_vector(self.sizes());
  outputSize[self.dim()-1] = output_size[1];
  outputSize[self.dim()-2] = output_size[0];

  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  NPUNativeFunctions::adaptive_avg_pool2d_out(self, output_size, result);

  return result;
}


} // namespace native
} // namespace at_npu
