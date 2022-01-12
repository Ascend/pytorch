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

Tensor& binary_cross_entropy_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  Tensor weightTensor = weight;
  if (!weight.defined()) {
    weightTensor = at::ones(self.sizes(), self.options());
  }

  // constructs the attr of the NPUAttrDesc
  std::string reductionStr = NpuUtils::get_reduction_str(reduction);

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("BinaryCrossEntropy")
      .Input(self)
      .Input(target)
      .Input(weightTensor)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();

  return result;
}

Tensor binary_cross_entropy_npu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  // calculate the output size
  IntArrayRef outputSize;

  if (reduction == Reduction::None) {
    outputSize = input_same_output_size(self);
  } else {
    outputSize = ArrayRef<int64_t>();
  }

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  if (self.numel() == 0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = result.to(at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }

  // calculate the output result of the NPU
  binary_cross_entropy_out_npu(result, self, target, weight, reduction);

  return result;
}

} // namespace native
} // namespace at
