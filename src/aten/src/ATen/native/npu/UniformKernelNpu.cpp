// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& uniform_out_npu(
    Tensor& result,
    const Tensor& self,
    double from,
    double to,
    Generator* gen_) {
  OpCommand cmd;
  cmd.Name("Uniform")
    .Input(self)
    .Output(result)
    .Attr("from", static_cast<float>(from))
    .Attr("to", static_cast<float>(to))
    .Run();

  return result;
}

Tensor& uniform_npu_(Tensor& self, double from, double to, Generator* gen_) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  // TODO: The operator needs to use fp32 for calculation.
  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = self.to(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&selfCopy)) {
    Tensor selfContiguous = NpuUtils::format_contiguous(selfCopy);
    Tensor result =
        uniform_out_npu(selfContiguous, selfContiguous, from, to, gen_);
    NpuUtils::format_fresh_view(selfCopy, result);
  } else {
    uniform_out_npu(selfCopy, selfCopy, from, to, gen_);
  }
  self.copy_(selfCopy);
  return self;
}

} // namespace native
} // namespace at