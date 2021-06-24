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

Tensor prelu_npu(const Tensor& self, const Tensor& weight_) {
  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  // calculate the output size
  auto outputSize = input_same_output_size(self);
  Tensor result = at::empty_with_format(
  outputSize, input.options(), CalcuOpUtil::get_tensor_npu_format(input));
  
  OpCommand cmd;
  cmd.Name("PRelu")
     .Input(self)
     .Input(weight)
     .Output(result)
     .Run();
  return result;
}

} // namespace native
} // namespace at
