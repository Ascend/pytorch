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

Tensor& selu_out_npu(Tensor& result, const Tensor& self) {
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Selu")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

Tensor selu_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  selu_out_npu(result, self);

  return result;
}

Tensor& selu_npu_(Tensor& self) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = selu_out_npu(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    selu_out_npu(self, self);
  }

  return self;
}

} // namespace native
} // namespace at