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

SmallVector<NPUTensorDesc, N> nms_with_mask_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> nms_with_mask_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> nms_with_mask_npu_attr(Scalar iou_threshold) {
  float iouThresholdValue = CalcuOpUtil::get_scalar_float_value(iou_threshold);
  NPUAttrDesc npuAttrIouThreshold =
      NPUAttrDesc("iou_threshold", iouThresholdValue);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrIouThreshold};

  return attrs;
}

tuple<Tensor&, Tensor&, Tensor&> nms_with_mask_out_npu(
    Tensor& boxes,
    Tensor& idx,
    Tensor& mask,
    const Tensor& input,
    Scalar iou_threshold) {
  // constructs the input and output NPUTensorDesc
  auto inputs = nms_with_mask_npu_input({input});
  auto outputs = nms_with_mask_npu_output({boxes, idx, mask});

  // constructs the attr of the NPUAttrDesc
  auto attrs = nms_with_mask_npu_attr(iou_threshold);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("NMSWithMask", inputs, outputs, attrs);

  return std::tuple<Tensor&, Tensor&, Tensor&>(boxes, idx, mask);
}

tuple<Tensor, Tensor, Tensor> nms_with_mask_npu(
    const Tensor& input,
    Scalar iou_threshold) {
  // calculate the output size
  auto outputSizes = nms_with_mask_npu_output_size(input);

  // construct the output tensor of the NPU
  Tensor boxes = at::empty_with_format(
      std::get<0>(outputSizes),
      input.options(),
      CalcuOpUtil::get_tensor_npu_format(input));

  Tensor idx = at::empty_with_format(
      std::get<1>(outputSizes),
      input.options().dtype(at::kInt),
      CalcuOpUtil::get_tensor_npu_format(input));

  Tensor mask = at::empty_with_format(
      std::get<2>(outputSizes),
      input.options().dtype(at::kByte),
      CalcuOpUtil::get_tensor_npu_format(input));

  nms_with_mask_out_npu(boxes, idx, mask, input, iou_threshold);

  return std::tuple<Tensor, Tensor, Tensor>(boxes, idx, mask);
}

} // namespace native
} // namespace at
