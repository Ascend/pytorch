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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> roi_align_backward_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> roi_align_backward_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> roi_align_backward_npu_attr(
    IntArrayRef xdiff_shape,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sample_num) {
  NPUAttrDesc xdiffshapeValue = NPUAttrDesc("xdiff_shape", xdiff_shape);
  NPUAttrDesc spatialscaleValue =
      NPUAttrDesc("spatial_scale", (float)spatial_scale);
  NPUAttrDesc pooledheightValue = NPUAttrDesc("pooled_height", pooled_height);
  NPUAttrDesc pooledwidthValue = NPUAttrDesc("pooled_width", pooled_width);
  NPUAttrDesc samplenumValue = NPUAttrDesc("sample_num", sample_num);

  SmallVector<NPUAttrDesc, N> attrs = {xdiffshapeValue,
                                       spatialscaleValue,
                                       pooledheightValue,
                                       pooledwidthValue,
                                       samplenumValue};
  return attrs;
}

Tensor& roi_align_backward_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& rois,
    IntArrayRef xdiff_shape,
    int64_t pooled_width,
    int64_t pooled_height,
    double spatial_scale,
    int64_t sample_num) {
  // constructs the input and output NPUTensorDesc
  auto inputs = roi_align_backward_npu_input({self, rois});
  auto outputs = roi_align_backward_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = roi_align_backward_npu_attr(
      xdiff_shape, spatial_scale, pooled_height, pooled_width, sample_num);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("ROIAlignGrad", inputs, outputs, attrs);

  return result;
}

Tensor roi_align_backward_npu(
    const Tensor& self,
    const Tensor& rois,
    IntArrayRef xdiff_shape,
    int64_t pooled_width,
    int64_t pooled_height,
    double spatial_scale,
    int64_t sample_num) {
  // calculate the output size
  auto outputSize = roi_align_backward_npu_output_size(xdiff_shape);

  // construct the output tensor of the NPU
  Tensor result =
      at::empty_with_format(outputSize, self.options(), ACL_FORMAT_NC1HWC0);

  //Check the self empty
  for (int i = 0; i < self.dim(); i++) {
      if (self.size(i) == 0) {
          result.fill_(0);
          return result;
      }
  }

  // calculate the output result of the NPU
  roi_align_backward_out_npu(
      result,
      self,
      rois,
      xdiff_shape,
      pooled_width,
      pooled_height,
      spatial_scale,
      sample_num);

  return result;
}

} // namespace native
} // namespace at