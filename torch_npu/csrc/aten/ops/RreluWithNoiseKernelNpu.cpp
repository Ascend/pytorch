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

#include <ATen/core/DistributionsHelper.h>

namespace at_npu {
namespace native {

void _rrelu_with_noise_train(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& noise,
    at::Scalar lower_,
    at::Scalar upper_,
    c10::optional<at::Generator> generator) {
  // use vector calculation instead of point-loop calculation
  double lower = lower_.toDouble();
  double upper = upper_.toDouble();
  at::Tensor uniform_tensor = at::empty(input.sizes(), input.options()).uniform_(lower, upper, generator);
  at::Tensor mask_tensor = input.le(0);
  at::Tensor one_tensor = at::empty(input.sizes(), input.options()).fill_(1).to(noise.dtype());
  at::Tensor select_tensor = at::_s_where(mask_tensor, uniform_tensor, one_tensor);
  noise.copy_(select_tensor);
  at::Tensor result = output.contiguous();
  result = input.mul(noise);
  output.copy_(result);
}

at::Tensor& rrelu_with_noise_out_nocheck(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator,
    at::Tensor& output) {
      
  if (training) {
    _rrelu_with_noise_train(output, self.contiguous(), noise, lower, upper, generator);
    return output;
  } else {
    auto float_lower = lower.toFloat();
    auto float_upper = upper.toFloat();
    at::Scalar negative_slope = (float_lower + float_upper) / 2;
    return NPUNativeFunctions::leaky_relu_out(self, negative_slope, output);
  }
}

at::Tensor NPUNativeFunctions::rrelu_with_noise(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator) {
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return rrelu_with_noise_out_nocheck(self, noise, lower, upper, training, generator, output);
}

at::Tensor& NPUNativeFunctions::rrelu_with_noise_(
    at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator) {
  return NPUNativeFunctions::rrelu_with_noise_out(self, noise, lower, upper, training, generator, self);
}

at::Tensor& NPUNativeFunctions::rrelu_with_noise_out(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator,
    at::Tensor& output) {
  OpPreparation::CheckOut(
      {self, noise},
      output,
      self);
  
  if (!NpuUtils::check_match(&output)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(output);
    at::Tensor newResult = rrelu_with_noise_out_nocheck(self, noise, lower, upper, training, generator, contiguousResult);
    NpuUtils::format_fresh_view(output, newResult);
  } else {
    rrelu_with_noise_out_nocheck(self, noise, lower, upper, training, generator, output);
  }

  return output;
}
} // namespace native
} // namespace at_npu