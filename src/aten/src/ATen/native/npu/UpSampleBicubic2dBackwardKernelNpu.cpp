// Copyright (c) 2020, Huawei Technologies.
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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include <vector>

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> upsample_bicubic2d_backward_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> upsample_bicubic2d_backward_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> upsample_bicubic2d_backward_npu_attr(
    IntArrayRef original_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  float temp_h = 0.0;
  float temp_w = 0.0;
  if(scales_h.has_value()) {
    temp_h = (float)scales_h.value();
  } 
  if(scales_w.has_value()) {
    temp_w = (float)scales_w.value();
  }
  SmallVector<float,N> scales = {temp_h, temp_w};
  NPUAttrDesc npuAttrScales = NPUAttrDesc("scales", scales);
  SmallVector<float, N> roi = {};
  NPUAttrDesc npuAttrRoi = NPUAttrDesc("roi", roi);
  NPUAttrDesc npuAttrOriginalSize = NPUAttrDesc("original_size", original_size);
  string coordinate_transformation_mode =
      align_corners ? "align_corners" : "half_pixel";
  NPUAttrDesc npuAttrCoordinateTransformationMode = 
      NPUAttrDesc("coordinate_transformation_mode", coordinate_transformation_mode);

  //cubic_coeff_a
  //The coefficient 'a' used in cubic interpolation.
  //Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch).
  //This attribute is valid only if "mode" is "cubic".
  float cu = -0.75;
  NPUAttrDesc npuAttrCubicCoeffA = NPUAttrDesc("cubic_coeff_a", cu);

  //exclude_outside
  //If set to 1, the weight of sampling locations outside the tensor will be set to 0
  //and the weight will be renormalized so that their sum is 1.0. The default value is 0.
  int64_t ex = 0;
  NPUAttrDesc npuAttrExcludeOutside = NPUAttrDesc("exclude_outside", ex);

  //extrapolation_value
  //When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside
  //the range [0, length_original - 1], this value is used as the corresponding output value.
  //Default is 0.0f.
  float ext = 0.0;
  NPUAttrDesc npuAttrExtrapolationValue = NPUAttrDesc("extrapolation_value", ext);
  string mode = "cubic";
  NPUAttrDesc npuAttrMode = NPUAttrDesc("mode", mode);
  string ne = "round_prefer_floor";
  NPUAttrDesc npuAttrNearestMode = NPUAttrDesc("nearest_mode", ne);
  SmallVector<NPUAttrDesc, N> attrs = 
  {
    npuAttrOriginalSize,
    npuAttrRoi,
    npuAttrScales,
    npuAttrCoordinateTransformationMode,
    npuAttrCubicCoeffA,
    npuAttrExcludeOutside,
    npuAttrExtrapolationValue,
    npuAttrMode,
    npuAttrNearestMode,
  };
  return attrs;
}

Tensor& upsample_bicubic2d_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output, 
    IntArrayRef output_size, 
    IntArrayRef input_size, 
    bool align_corners, 
    c10::optional<double> scales_h,
    c10::optional<double> scales_w
  ) {
  // constructs the input and output NPUTensorDesc

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  auto inputs = upsample_bicubic2d_backward_npu_input({grad_output});
  auto outputs = upsample_bicubic2d_backward_npu_output({grad_input});
  
  // constructs the attr of the NPUAttrDesc
  auto attrs = upsample_bicubic2d_backward_npu_attr(input_size, align_corners, scales_h, scales_w);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("ResizeGradD", inputs, outputs, attrs);
  return grad_input;
}

Tensor upsample_bicubic2d_backward_npu(
    const Tensor& grad_output, 
    IntArrayRef output_size, 
    IntArrayRef input_size, 
    bool align_corners, 
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // construct the output tensor of the NPU
  auto outputSize = upsample_bicubic2d_backward_npu_output_size(input_size);
  Tensor result = at::empty_with_format(
      outputSize, grad_output.options(), CalcuOpUtil::get_tensor_npu_format(grad_output));
  // calculate the output result of the NPU
  return upsample_bicubic2d_backward_out_npu(result, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
}//namespace native
}//namespace at