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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& adaptive_avg_pool3d_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef output_size) {
  // reuse the mean out when d,h,w=1
  if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1) {
    at::mean_out(result, self, {self.dim() - 3, self.dim() - 2, self.dim() - 1}, true);
  } else {
    TORCH_CHECK(false,
        "adaptive_avg_pool3d only support D=1 && H=1 && W=1 current!");
  }
  return result;
}

Tensor adaptive_avg_pool3d_npu(const Tensor& self, IntArrayRef output_size) {
  for (int64_t i = 0; i < self.dim(); i++) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pooling3d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (self.dim() == 4 || self.dim() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  auto outputSize = adaptive_avg_pool3d_npu_output_size(self, output_size);
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  adaptive_avg_pool3d_out_npu(result, self, output_size);
  return result;
}

} // namespace native
} // namespace at
