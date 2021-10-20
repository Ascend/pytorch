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

Tensor& upsample_nearest2d_backward_out_npu(
    const Tensor& grads,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& y) {
  SmallVector<int64_t, N> outputSize = {input_size[2], input_size[3]};
  OpCommand cmd;
  cmd.Name("ResizeNearestNeighborV2Grad")
      .Input(grads)
      .Input(outputSize, at::kInt)
      .Output(y)
      .Attr("align_corners", false)
      .Attr("half_pixel_centers", false)
      .Run();

   return y;
}

Tensor upsample_nearest2d_backward_npu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor grads = grad_output;
  if (grad_output.scalar_type() != at::ScalarType::Float) {
    grads = grad_output.to(at::kFloat);
  }
  Tensor grad_input = OpPreparation::ApplyTensor(
      input_size, grads.options(), grad_output);
  upsample_nearest2d_backward_out_npu(
      grads, output_size, input_size, scales_h, scales_w, grad_input);
  return grad_input;
}

Tensor upsample_nearest2d_backward_vec_npu(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input_size, output_size, scale_factors);
  auto scales_h = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 1);
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output, input_size);
  upsample_nearest2d_backward_out_npu(grad_output, osize, input_size, scales_h, scales_w, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("upsample_nearest2d_backward.vec", TORCH_FN(upsample_nearest2d_backward_vec_npu));
  m.impl("upsample_nearest2d_backward", TORCH_FN(upsample_nearest2d_backward_npu));
  m.impl("upsample_nearest2d_backward.out", TORCH_FN(upsample_nearest2d_backward_out_npu));
}

} // namespace native
} // namespace at