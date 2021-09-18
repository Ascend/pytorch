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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& elu_out_npu(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale, Tensor& result) {
    OpPreparation::CheckMemory({self}, {result});
    float alphaValue = CalcuOpUtil::get_scalar_float_value(alpha);
    float scaleValue = CalcuOpUtil::get_scalar_float_value(scale);
    float inputScaleValue = CalcuOpUtil::get_scalar_float_value(input_scale);
  OpCommand cmd;
  cmd.Name("Elu")
    .Input(self)
    .Output(result)
    .Attr("alpha", alphaValue)
    .Attr("scale", scaleValue)
    .Attr("input_scale", inputScaleValue)
    .Run();
  return result;
}

Tensor elu_npu(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
  Tensor result = OpPreparation::ApplyTensor(self);

  elu_out_npu(self, alpha, scale, input_scale, result);
  return result;
}

Tensor& elu_npu_(Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = elu_out_npu(contiguousSelf, alpha, scale, input_scale, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    elu_out_npu(self, alpha, scale, input_scale, self);
  }
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("elu", TORCH_FN(elu_npu));
  m.impl("elu.out", TORCH_FN(elu_out_npu));
  m.impl("elu_", TORCH_FN(elu_npu_));
}
} // namespace native
} // namespace at
