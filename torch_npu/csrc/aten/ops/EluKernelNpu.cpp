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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& elu_out_nocheck(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, at::Tensor& result) {
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

at::Tensor& NPUNativeFunctions::elu_out(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor checkResult = elu_out_nocheck(self, alpha, scale, input_scale, contiguousResult);
    NpuUtils::format_fresh_view(result, checkResult);
  } else {
    elu_out_nocheck(self, alpha, scale, input_scale, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::elu(const at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  elu_out_nocheck(self, alpha, scale, input_scale, result);
  return result;
}

at::Tensor& NPUNativeFunctions::elu_(at::Tensor& self, at::Scalar alpha, at::Scalar scale, at::Scalar input_scale) {
  elu_out(self, alpha, scale, input_scale, self);
  return self;
}
} // namespace native
} // namespace at_npu
