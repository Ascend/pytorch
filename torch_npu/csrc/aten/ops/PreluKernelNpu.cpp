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

at::Tensor NPUNativeFunctions::prelu(const at::Tensor& self, const at::Tensor& weight_) {
  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  // calculate the output size
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(input, outputSize);
  
  OpCommand cmd;
  cmd.Name("PRelu")
     .Input(self)
     .Input(weight)
     .Output(result)
     .Run();
  return result;
}
} // namespace native
} // namespace at_npu