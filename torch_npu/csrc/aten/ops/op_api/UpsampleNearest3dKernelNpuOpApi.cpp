// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> upsample_nearest3d_npu_output_size(
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

  at::SmallVector<int64_t, SIZE> outputSize = 
    {nbatch, channels, output_depth, output_height, output_width};
  
  return outputSize;
}

at::Tensor& NPUNativeOpApiFunctions::upsample_nearest3d_out(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnUpsampleNearest3d,
                   NPUNativeFunctions::upsample_nearest3d_out(input, output_size,scales_d, scales_h, scales_w,
                                                              result));
  at::SmallVector<int64_t, SIZE> output_osize = upsample_nearest3d_npu_output_size(input, output_size,
                                                                                   scales_d, scales_h, scales_w);
  OpPreparation::check_tensor({input}, result, input, output_osize);
  double scales_d_attr = scales_d.value_or(0);
  double scales_h_attr = scales_h.value_or(0);
  double scales_w_attr = scales_w.value_or(0);
  EXEC_NPU_CMD(aclnnUpsampleNearest3d, input, output_size, scales_d_attr, scales_h_attr, scales_w_attr, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_nearest3d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  DO_COMPATIBILITY(aclnnUpsampleNearest3d,
                   NPUNativeFunctions::upsample_nearest3d(input, output_size, scales_d, scales_h, scales_w));
  at::SmallVector<int64_t, SIZE> output_osize = upsample_nearest3d_npu_output_size(input, output_size,
                                                                                   scales_d, scales_h, scales_w);
  at::Tensor result = OpPreparation::apply_tensor_without_format(input, output_osize);
  double scales_d_attr = scales_d.value_or(0);
  double scales_h_attr = scales_h.value_or(0);
  double scales_w_attr = scales_w.value_or(0);
  EXEC_NPU_CMD(aclnnUpsampleNearest3d, input, output_size, scales_d_attr, scales_h_attr, scales_w_attr, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_nearest3d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  DO_COMPATIBILITY(aclnnUpsampleNearest3d,
                   NPUNativeFunctions::upsample_nearest3d(input, output_size, scale_factors));
  auto osize = CalcuOpUtil::ComputeOutputSize(input.sizes(), output_size, scale_factors);
  auto scales_d = CalcuOpUtil::GetScaleValue(scale_factors, 0);
  auto scales_h = CalcuOpUtil::GetScaleValue(scale_factors, 1);
  auto scales_w = CalcuOpUtil::GetScaleValue(scale_factors, 2);
  double scales_d_attr = scales_d.value_or(0);
  double scales_h_attr = scales_h.value_or(0);
  double scales_w_attr = scales_w.value_or(0);
  at::SmallVector<int64_t, SIZE> output_size_vec = upsample_nearest3d_npu_output_size(input, osize,
                                                                                      scales_d, scales_h, scales_w);
  at::Tensor result = OpPreparation::apply_tensor_without_format(input, output_size_vec);
  auto output_osize = at::IntArrayRef(osize);
  EXEC_NPU_CMD(aclnnUpsampleNearest3d, input, output_osize, scales_d_attr, scales_h_attr, scales_w_attr, result);
  return result;
}

} // namespace native
} // namespace at_npu

