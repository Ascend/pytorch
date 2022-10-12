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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
c10::SmallVector<int64_t, SIZE> upsample_nearest1d_npu_output_size(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  c10::SmallVector<int64_t, SIZE> outputSize;
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t W = output_size[0];
  outputSize = {N, C, 1, W};
  return outputSize;
}

at::Tensor& upsample_nearest1d_out_nocheck(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales,
    at::Tensor& result) {
  TORCH_CHECK(
      (self.size(1) != 0 && self.size(2) != 0) && self.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      self.sizes());

  at::Tensor selfOp = self.unsqueeze(2);
  OpCommand cmd;
  cmd.Name("Resize")
      .Input(selfOp)
      .Input(output_size, at::kFloat)
      .Input(output_size, at::kFloat)
      .Input(result.sizes(), at::kLong)
      .Output(result)
      .Attr("mode", (string)"nearest");
  if (self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf) {
    cmd.Attr("nearest_mode", (string)"round_prefer_floor")
        .Attr("coordinate_transformation_mode", (string)"half_pixel")
        .Run();
  } else {
    cmd.Attr("nearest_mode", (string)"floor")
        .Attr("coordinate_transformation_mode", (string)"pytorch_half_pixel")
        .Run();
  }
  result = result.squeeze(2);
  return result;
}

at::Tensor& NPUNativeFunctions::upsample_nearest1d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales,
    at::Tensor& result) {
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_npu_output_size(self, output_size);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    upsample_nearest1d_out_nocheck(self, output_size, scales, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    upsample_nearest1d_out_nocheck(self, output_size, scales, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::upsample_nearest1d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input.sizes(), output_size, scale_factors);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 0);
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_npu_output_size(input, osize);
  at::Tensor result = OpPreparation::ApplyTensor(input, outputSize);

  upsample_nearest1d_out_nocheck(input, osize, scales_w, result);
  return result;
}

at::Tensor NPUNativeFunctions::upsample_nearest1d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales) {
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_npu_output_size(self, output_size);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  upsample_nearest1d_out_nocheck(self, output_size, scales, result);
  return result;
}
} // namespace native
} // namespace at_npu
