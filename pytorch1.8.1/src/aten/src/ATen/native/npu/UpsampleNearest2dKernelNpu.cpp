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

SmallVector<int64_t, SIZE> upsample_nearest2d_npu_output_size(
    const Tensor& input,
    IntArrayRef output_size){
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};

  return outputSize;
}

Tensor& upsample_nearest2d_out_npu(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
  SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);
  if (!result.sizes().equals(outputSize)){
    result.resize_(outputSize);
  }
  SmallVector<int64_t,N> outputSizeVec = array_to_small_vector(output_size);
  OpCommand cmd;
  cmd.Name("ResizeNearestNeighborV2")
    .Input(self)
    .Input(outputSizeVec, at::kInt)
    .Output(result)
    .Attr("align_corners", false)
    .Attr("half_pixel_centers", false)
    .Run();
  return result;
}

Tensor upsample_nearest2d_npu(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_npu_output_size(self, output_size);

  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  upsample_nearest2d_out_npu(self, output_size, scales_h, scales_w, result);

  return result;
}

Tensor upsample_nearest2d_vec_npu(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scale_w = CalcuOpUtil::get_scale_value(scale_factors, 1);
  Tensor result = OpPreparation::ApplyTensor(input, osize);
  upsample_nearest2d_out_npu(input, osize, scale_h, scale_w, result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("upsample_nearest2d.vec", TORCH_FN(upsample_nearest2d_vec_npu));
  m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d_npu));
  m.impl("upsample_nearest2d.out", TORCH_FN(upsample_nearest2d_out_npu));
}

} // namespace native
} // namespace at