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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::upsample_bilinear2d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackward,
                   NPUNativeFunctions::upsample_bilinear2d_backward_out(grad_output, output_size, input_size,
                                                                        align_corners, scales_h, scales_w, grad_input));

  OpPreparation::CheckOut({grad_output}, grad_input, grad_output, input_size);
  double scales_h_attr = scales_h.value_or(1);
  double scales_w_attr = scales_w.value_or(1);
  EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackward, grad_output, output_size, input_size, align_corners,
               scales_h_attr, scales_w_attr, grad_input);
  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::upsample_bilinear2d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackward,
                   NPUNativeFunctions::upsample_bilinear2d_backward(grad_output, output_size, input_size,
                                                                    align_corners, scales_h, scales_w));
  auto outputSize = upsample_bilinear2d_backward_npu_output_size(
      grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output, outputSize);

  double scales_h_attr = scales_h.value_or(1);
  double scales_w_attr = scales_w.value_or(1);

  EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackward, grad_output, output_size, input_size, align_corners,
               scales_h_attr, scales_w_attr, grad_input);
  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::upsample_bilinear2d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackward,
                   NPUNativeFunctions::upsample_bilinear2d_backward(grad_output, output_size, input_size,
                                                                    align_corners, scale_factors));
  auto osize = CalcuOpUtil::ComputeOutputSize(input_size, output_size, scale_factors);
  auto scales_h = CalcuOpUtil::GetScaleValue(scale_factors, 0);
  auto scales_w = CalcuOpUtil::GetScaleValue(scale_factors, 1);
  double scales_h_attr = scales_h.value_or(1);
  double scales_w_attr = scales_w.value_or(1);
  
  auto outputsize = at::IntArrayRef(osize);
  auto outputSize = upsample_bilinear2d_backward_npu_output_size(
      grad_output, osize, input_size, align_corners, scales_h, scales_w);
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output, outputSize);

  EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackward, grad_output, outputsize, input_size, align_corners,
               scales_h_attr, scales_w_attr, grad_input);
  return grad_input;
}
} // namespace native
} // namespace at_npu

