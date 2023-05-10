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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::rrelu_with_noise_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self_or_result,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    bool is_result) {
  auto minimum = 1E-6;
  auto folat_lower = lower.toFloat();
  auto float_upper = upper.toFloat();
  if (training && (float_upper - folat_lower > minimum)) {
    return grad_output.mul(noise);
  } else {
    at::Scalar negative_slope = (folat_lower + float_upper) / 2;
    return NPUNativeFunctions::leaky_relu_backward(grad_output, self_or_result, negative_slope, is_result);
  }
}

} // namespace native
} // namespace at_npu