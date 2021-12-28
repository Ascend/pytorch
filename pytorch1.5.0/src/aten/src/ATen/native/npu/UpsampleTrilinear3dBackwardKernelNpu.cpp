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

SmallVector<int64_t, SIZE> upsample_trilinear3d_backward_outputsize_npu(
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

  SmallVector<int64_t, SIZE> outputSize = 
    {nbatch, channels, input_depth, input_height, input_width};
  return outputSize;
}

Tensor& upsample_trilinear3d_backward_npu_nocheck(
    Tensor& out,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_output_copy = (grad_output.scalar_type() == at::kHalf) ?
    grad_output.npu_dtype_cast(at::kFloat) : grad_output;

  OpCommand cmd;
  cmd.Name("UpsampleTrilinear3dGrad")
    .Input(grad_output_copy)
    .Output(out)
    .Attr("input_size", input_size)
    .Attr("output_size", output_size)
    .Attr("align_corners", align_corners)
    .Run();

  return out;
}

Tensor& upsample_trilinear3d_backward_out_npu(
    Tensor& out,
    const Tensor& grad,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto outputsize = upsample_trilinear3d_backward_outputsize_npu(
      output_size, input_size, scales_d, scales_h, scales_w);

  OpPreparation::CheckOut({grad}, out, grad, outputsize);

  if (grad.scalar_type() == at::kHalf) {
    Tensor out_copy = OpPreparation::ApplyTensorWithSizes(
        outputsize, grad.options().dtype(at::kFloat));
    
    upsample_trilinear3d_backward_npu_nocheck(
        out_copy, grad, output_size, input_size, align_corners, scales_d, scales_h, scales_w);

    out_copy = out_copy.npu_dtype_cast(grad.scalar_type());
    NpuUtils::format_fresh_view(out, out_copy);
  } else if (!NpuUtils::check_match(&out)) {
    auto contiguous_out = NpuUtils::format_contiguous(out);

    upsample_trilinear3d_backward_npu_nocheck(
        out, grad, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    
    NpuUtils::format_fresh_view(out, contiguous_out);   
  } else {
    upsample_trilinear3d_backward_npu_nocheck(
        out, grad, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  }

  return out;
}

Tensor upsample_trilinear3d_backward_npu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto outputsize = upsample_trilinear3d_backward_outputsize_npu(
      output_size, input_size, scales_d, scales_h, scales_w);

  Tensor result = (grad_output.scalar_type() == at::kHalf) ?
    OpPreparation::ApplyTensorWithSizes(outputsize, grad_output.options().dtype(at::kFloat)) :
    OpPreparation::ApplyTensor(grad_output, outputsize);

  upsample_trilinear3d_backward_npu_nocheck(
      result, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);

  if (grad_output.scalar_type() == at::kHalf) {
      result = result.npu_dtype_cast(at::kHalf);
  }

  return result;
}

} // namespace native
} // namespace at
