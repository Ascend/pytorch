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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& expm1_out_npu(Tensor& out, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("Expm1")
      .Input(self)
      .Output(out)
      .Run();
  return out;
}

Tensor expm1_npu(const Tensor& self) {
  auto outputSize = input_same_output_size(self); 
  Tensor result = at::empty_with_format(outputSize, 
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));
  expm1_out_npu(result, self);
  return result;
}

Tensor& expm1_npu_(Tensor& self) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = expm1_out_npu(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    expm1_out_npu(self, self);
  }
  return self;
}
}
}
