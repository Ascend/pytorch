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

Tensor& upsample_nearest3d_backward_out_npu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
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

  OpCommand cmd;
  cmd.Name("UpsampleNearest3dGrad")
    .Input(grad_output)
    .Output(grad_input)
    .Attr("input_size", input_size)
    .Attr("output_size", output_size)
    .Run();

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
  upsample_nearest3d_backward_out_npu(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
  return grad_input;
}

Tensor upsample_nearest3d_backward_vec_npu(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input_size, output_size, scale_factors);
  auto scales_d = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scales_h = CalcuOpUtil::get_scale_value(scale_factors, 1);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 2);
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output, input_size);
  upsample_nearest3d_backward_out_npu(grad_output, osize, input_size, scales_d, scales_h, scales_w, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("upsample_nearest3d_backward.vec", TORCH_FN(upsample_nearest3d_backward_vec_npu));
  m.impl("upsample_nearest3d_backward", TORCH_FN(upsample_nearest3d_backward_npu));
  m.impl("upsample_nearest3d_backward.out", TORCH_FN(upsample_nearest3d_backward_out_npu));
}

} // namespace native
} // namespace at