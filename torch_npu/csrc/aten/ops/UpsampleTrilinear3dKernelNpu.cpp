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

at::SmallVector<int64_t, SIZE> upsample_trilinear3d_outputsize_npu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);

  at::SmallVector<int64_t, SIZE> outputSize = 
    {nbatch, channels, output_depth, output_height, output_width};
  return outputSize;
}

at::Tensor& upsample_trilinear3d_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at::Tensor input_copy = (input.scalar_type() == at::ScalarType::Half) ?
    NPUNativeFunctions::npu_dtype_cast(input, at::kFloat) : input;
    
  OpCommand cmd;
  cmd.Name("UpsampleTrilinear3d")
    .Input(input_copy)
    .Output(result)
    .Attr("output_size", output_size)
    .Attr("align_corners", align_corners)
    .Run();
  
  return result;
}

at::Tensor& NPUNativeFunctions::upsample_trilinear3d_out(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result) {
  auto outputsize = upsample_trilinear3d_outputsize_npu(
      input, output_size, scales_d, scales_h, scales_w);

  OpPreparation::CheckOut({input}, result, input, outputsize);
  
  if (input.scalar_type() == at::kHalf) {
    auto out_copy = OpPreparation::ApplyTensorWithSizes(
        outputsize, input.options().dtype(at::kFloat));
    
    upsample_trilinear3d_npu_nocheck(
        out_copy, input, output_size, align_corners, scales_d, scales_h, scales_w);

    out_copy = NPUNativeFunctions::npu_dtype_cast(out_copy, input.scalar_type());
    NpuUtils::format_fresh_view(result, out_copy);
  } else if (!NpuUtils::check_match(&result)) {
    auto contiguous_out = NpuUtils::format_contiguous(result);

    upsample_trilinear3d_npu_nocheck(
        contiguous_out, input, output_size, align_corners, scales_d, scales_h, scales_w);

    NpuUtils::format_fresh_view(result, contiguous_out);
  } else {
    upsample_trilinear3d_npu_nocheck(
        result, input, output_size, align_corners, scales_d, scales_h, scales_w);
  }

  return result;
}

at::Tensor NPUNativeFunctions::upsample_trilinear3d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto outputsize = upsample_trilinear3d_outputsize_npu(
      input, output_size, scales_d, scales_h, scales_w);

  at::Tensor result = (input.scalar_type() == at::kHalf) ?
    OpPreparation::ApplyTensorWithSizes(outputsize, input.options().dtype(at::kFloat)) :
    OpPreparation::ApplyTensor(input, outputsize);

  upsample_trilinear3d_npu_nocheck(
      result, input, output_size, align_corners, scales_d, scales_h, scales_w);
  
  if (input.scalar_type() == at::kHalf) {
      result = NPUNativeFunctions::npu_dtype_cast(result, input.scalar_type());
  }

  return result;
}

at::Tensor NPUNativeFunctions::upsample_trilinear3d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input.sizes(), output_size, scale_factors);
  auto scales_d = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scales_h = CalcuOpUtil::get_scale_value(scale_factors, 1);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 2);
  at::Tensor result = NPUNativeFunctions::upsample_trilinear3d(
      input, osize, align_corners, scales_d, scales_h, scales_w);
  return result;
}

} // namespace native
} // namespace at_npu
