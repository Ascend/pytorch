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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& upsample_nearest3d_backward_out_npu_nocheck(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_depth = input_size[2];
  int64_t input_height = input_size[3];
  int64_t input_width = input_size[4];

  grad_input.resize_(
      {nbatch, channels, input_depth, input_height, input_width});
  
  Tensor gradOutputCopy = grad_output;
  Tensor gradInputCopy = grad_input;
  if (grad_output.scalar_type() == ScalarType::Half) {
    gradOutputCopy = gradOutputCopy.npu_dtype_cast(ScalarType::Float);
    gradInputCopy = gradInputCopy.npu_dtype_cast(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("UpsampleNearest3dGrad")
    .Input(gradOutputCopy)
    .Output(gradInputCopy)
    .Attr("input_size", input_size)
    .Attr("output_size", output_size)
    .Run();

  if (grad_output.scalar_type() == ScalarType::Half) {
    gradInputCopy = gradInputCopy.npu_dtype_cast(ScalarType::Half);
  }
  grad_input.copy_(gradInputCopy);
  return grad_input;
}

Tensor& upsample_nearest3d_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      grad_output,
      input_size);
  
  if (!NpuUtils::check_match(&grad_input)) {
    Tensor contiguousGradInput = NpuUtils::format_contiguous(grad_input);
    upsample_nearest3d_backward_out_npu_nocheck(
        contiguousGradInput, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
    NpuUtils::format_fresh_view(grad_input, contiguousGradInput);
  } else {
    upsample_nearest3d_backward_out_npu_nocheck(
        grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
  }
  return grad_input;
}

Tensor upsample_nearest3d_backward_npu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  Tensor grad_input = OpPreparation::ApplyTensor(grad_output, input_size);

  upsample_nearest3d_backward_out_npu_nocheck(grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);

  return grad_input;
}

} // namespace native
} // namespace at