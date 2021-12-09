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

Tensor& upsample_trilinear3d_out_npu(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
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

  result.resize_({nbatch, channels, output_depth, output_height, output_width});

  OpCommand cmd;
  cmd.Name("UpsampleTrilinear3d")
    .Input(input)
    .Output(result)
    .Attr("output_size", output_size)
    .Attr("align_corners", align_corners)
    .Run();

  return result;
}

Tensor upsample_trilinear3d_vec_npu(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input.sizes(), output_size, scale_factors);
  auto scales_d = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scales_h = CalcuOpUtil::get_scale_value(scale_factors, 1);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 2);
  Tensor result = OpPreparation::ApplyTensor(input, {1});
  upsample_trilinear3d_out_npu(input, osize, align_corners, scales_d, scales_h, scales_w, result);
  return result;
}

Tensor upsample_trilinear3d_npu(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  Tensor result = OpPreparation::ApplyTensor(input, {1});

  upsample_trilinear3d_out_npu(input, output_size, align_corners, scales_d, scales_h, scales_w, result);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("upsample_trilinear3d.vec", TORCH_FN(upsample_trilinear3d_vec_npu));
  m.impl("upsample_trilinear3d", TORCH_FN(upsample_trilinear3d_npu));
  m.impl("upsample_trilinear3d.out", TORCH_FN(upsample_trilinear3d_out_npu));
}

} // namespace native
} // namespace at
