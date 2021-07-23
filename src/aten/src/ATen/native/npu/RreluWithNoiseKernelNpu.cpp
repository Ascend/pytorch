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
#include <ATen/core/DistributionsHelper.h>

namespace at {
namespace native {
using namespace at::native::npu;


void _rrelu_with_noise_train(
    Tensor& output,
    const Tensor& input,
    const Tensor& noise,
    Scalar lower_,
    Scalar upper_,
    Generator* generator) {
  float lower = lower_.toFloat();
  float upper = upper_.toFloat();
  auto shape = output.sizes();
  auto noise_shape = noise.sizes();
  Tensor tmp_tensor = output.contiguous();
  Tensor output_data = tmp_tensor.reshape({output.numel()});
  Tensor input_data = input.reshape({input.numel()});
  Tensor tmp_noise = noise;
  tmp_noise = tmp_noise.reshape({tmp_noise.numel()});
  auto gen = at::get_generator_or_default<CPUGenerator>(generator, detail::getDefaultCPUGenerator());

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

Tensor rrelu_with_noise_npu(
    const Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return rrelu_with_noise_out_npu(output, self, noise, lower, upper, training, generator);
}

Tensor& rrelu_with_noise_npu_(
    Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  return rrelu_with_noise_out_npu(self, self, noise, lower, upper, training, generator);
}

Tensor& rrelu_with_noise_out_npu(
    Tensor& output,
    const Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  
  if (training) {
    _rrelu_with_noise_train(output, self.contiguous(), noise, lower, upper, generator);
    return output;
  } else {
    auto float_lower = lower.toFloat();
    auto float_upper = upper.toFloat();
    Scalar negative_slope = (float_lower + float_upper) / 2;
    return at::leaky_relu_out(output, self, negative_slope);
  }
}


} // namespace native
} // namespace at
