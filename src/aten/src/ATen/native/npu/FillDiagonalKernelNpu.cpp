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

Tensor& fill_diagonal_out_npu(
    Tensor& result,
    const Tensor& self,
    Scalar& value,
    bool wrap) {
  float fill_value = CalcuOpUtil::get_scalar_float_value(value);
  OpCommand cmd;
  cmd.Name("FillDiagonal")
      .Input(self)
      .Output(result)
      .Attr("fill_value", fill_value)
      .Attr("wrap", wrap)
      .Run();

  return result;
}

Tensor& fill_diagonal_npu_(Tensor& self, Scalar value, bool wrap) {
  OpPreparation::CastBackToOriFormat(self);
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};

  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result =
        fill_diagonal_out_npu(contiguousSelf, contiguousSelf, value, wrap);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fill_diagonal_out_npu(self, self, value, wrap);
  }

  return self;
}

} // namespace native
} // namespace at