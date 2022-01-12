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
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& binary_cross_entropy_backward_out_npu(
    Tensor& gradInput,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  Tensor weightTensor = weight.defined() ? weight :
              at::ones(self.sizes(), self.options());

  std::string reductionStr = NpuUtils::get_reduction_str(reduction);
  OpCommand cmd;
  cmd.Name("BinaryCrossEntropyGrad")
     .Input(self)
     .Input(target)
     .Input(grad_output)
     .Input(weightTensor)
     .Output(gradInput)
     .Attr("reduction", reductionStr)
     .Run();

  return gradInput;
}

Tensor binary_cross_entropy_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  Tensor gradInput = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  binary_cross_entropy_backward_out_npu(
      gradInput, grad_output, self, target, weight, reduction);

  return gradInput;
}

} // namespace native
} // namespace at