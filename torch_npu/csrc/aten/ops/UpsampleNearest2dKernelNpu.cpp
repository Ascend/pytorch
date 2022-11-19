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
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> upsample_nearest2d_npu_output_size(
    const at::Tensor& input,
    at::IntArrayRef output_size){
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  at::SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};

  return outputSize;
}

at::Tensor& NPUNativeFunctions::upsample_nearest2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result) {
  at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);
  if (!result.sizes().equals(outputSize)){
    result.resize_(outputSize);
  }
  at::SmallVector<int64_t,N> outputSizeVec = array_to_small_vector(output_size);
  OpCommand cmd;
  cmd.Name("ResizeNearestNeighborV2")
    .Input(self, "x", ACL_FORMAT_NCHW)
    .Input(outputSizeVec, at::kInt)
    .Output(result, "y", ACL_FORMAT_NCHW)
    .Attr("align_corners", false)
    .Attr("half_pixel_centers", false)
    .Run();
  return result;
}

at::Tensor NPUNativeFunctions::upsample_nearest2d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  NPUNativeFunctions::upsample_nearest2d_out(self, output_size, scales_h, scales_w, result);

  return result;
}

at::Tensor NPUNativeFunctions::upsample_nearest2d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input.sizes(), output_size, scale_factors);
  at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(input, osize);
  auto scale_h = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scale_w = CalcuOpUtil::get_scale_value(scale_factors, 1);
  at::Tensor result = OpPreparation::ApplyTensor(input, outputSize);
  NPUNativeFunctions::upsample_nearest2d_out(input, osize, scale_h, scale_w, result);
  return result;
}

} // namespace native
} // namespace at_npu