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
  float lower = lower_.toFloat();
  float upper = upper_.toFloat();
  auto shape = output.sizes();
  auto noise_shape = noise.sizes();
  at::Tensor tmp_tensor = output.contiguous();
  at::Tensor output_data = tmp_tensor.reshape({output.numel()});
  at::Tensor input_data = input.reshape({input.numel()});
  at::Tensor tmp_noise = noise;
  tmp_noise = tmp_noise.reshape({tmp_noise.numel()});
  auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(generator, at::detail::getDefaultCPUGenerator());

  for (int64_t i = 0; i < input.numel(); i++) {
    if (input_data[i].item().toFloat() <= 0) {
      at::uniform_real_distribution<double> uniform(lower, upper);
      const float r = uniform(gen);
      output_data[i] = input_data[i] * r;
      tmp_noise[i] = r;
    } else {
      tmp_noise[i] = 1;
      output_data[i] = input_data[i];
    }
  }
  if (!output.is_contiguous()) {
    output.copy_(tmp_tensor);
  }
  tmp_noise.reshape(noise_shape);
  noise.copy_(tmp_noise);
  output.reshape(shape);
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