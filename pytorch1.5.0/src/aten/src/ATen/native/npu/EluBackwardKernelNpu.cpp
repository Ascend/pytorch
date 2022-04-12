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

#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& elu_backward_out_npu(Tensor& grad_input, const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor& output) {
  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      grad_output);
  float value = CalcuOpUtil::get_scalar_float_value(alpha);
  OpCommand cmd;
  cmd.Name("EluGradV2")
      .Input(grad_output)
      .Input(output)
      .Output(grad_input)
      .Attr("alpha", value)
      .Run();
  return grad_input;
}

Tensor elu_backward_npu(const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor& output) {
  Tensor result = OpPreparation::ApplyTensor(grad_output);
  elu_backward_out_npu(result, grad_output, alpha, scale, input_scale, output);
  return result;
}
} // namespace native
} // namespace at
