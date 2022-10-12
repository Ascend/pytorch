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

static inline void upsample_linear1d_check(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      (self.size(1) != 0 && self.size(2) != 0) && self.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      self.sizes());
  
  int64_t input_width = self.size(2);
  int64_t output_width = output_size[0];
  
  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");
}

at::Tensor& upsample_linear1d_out_nocheck(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales,
    at::Tensor& result) {
  upsample_linear1d_check(self, output_size);

  // Since only NCHW format input is currently supported, first convert the
  // input self (3 dimensions) to 4 dimensions as the input of npu
  at::Tensor selfcp = self.unsqueeze(2);

  // to calculate the value of scale
  c10::SmallVector<float, N> sc = {};
  if (scales.has_value()) {
    sc.push_back(scales.value());
  } else {
    float temp = float(output_size[0]) / float(selfcp.size(3));
    sc.push_back(temp);
  }

  string coordinate_transformation_mode =
      align_corners ? "align_corners" : "half_pixel";
  
  string mode = "linear";

  OpCommand cmd;
  cmd.Name("ResizeD")
      .Input(selfcp, "X", ACL_FORMAT_NCHW)
      .Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("sizes", output_size)
      .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
      .Attr("mode", mode)
      .Attr("scales", sc)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::upsample_linear1d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales,
    at::Tensor& result) {

  // calculate the output size
  auto outputSize = upsample_linear1d_npu_output_size(
      self, output_size, align_corners, scales);

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      outputSize);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor newResult = upsample_linear1d_out_nocheck(self, output_size, align_corners, scales, contiguousResult);
    NpuUtils::format_fresh_view(result, newResult);
  } else {
    upsample_linear1d_out_nocheck(self, output_size, align_corners, scales, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::upsample_linear1d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales) {
  // calculate the output size
  auto outputSize = upsample_linear1d_npu_output_size(
      self, output_size, align_corners, scales);
  
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  upsample_linear1d_out_nocheck(self, output_size, align_corners, scales, result);

  return result;
}

at::Tensor NPUNativeFunctions::upsample_linear1d(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(self.sizes(), output_size, scale_factors);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 0);
  // calculate the output size
  auto outputSize = upsample_linear1d_npu_output_size(
      self, osize, align_corners, scales_w);
  
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  upsample_linear1d_out_nocheck(self, osize, align_corners, scales_w, result);

  return result;
}

} // namespace native
} // namespace at_npu