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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& fill_diagonal_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& value,
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

at::Tensor& NPUNativeFunctions::fill_diagonal_(at::Tensor& self, const at::Scalar& value, bool wrap) {
  OpPreparation::CastBackToOriFormat(self);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result =
        fill_diagonal_out_npu(contiguousSelf, contiguousSelf, value, wrap);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fill_diagonal_out_npu(self, self, value, wrap);
  }

  return self;
}

} // namespace native
} // namespace at_npu