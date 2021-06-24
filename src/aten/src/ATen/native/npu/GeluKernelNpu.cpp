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

Tensor gelu_npu(const Tensor& self) {
  // calculate the output size
    auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
    Tensor result = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  // calculate the output result of the NPU
    OpCommand cmd;
    cmd.Name("Gelu")
        .Input(self)
        .Output(result)
        .Run();
    
    return result;
}

} // namespace native
} // namespace at
