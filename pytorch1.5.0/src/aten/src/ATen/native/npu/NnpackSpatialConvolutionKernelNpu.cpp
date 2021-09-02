// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

SmallVector<NPUTensorDesc, N> _nnpack_spatial_convolution_input(
    const SmallVector<Tensor, N>& input_tensor) {
  SmallVector<Tensor, N> inputTensors;
  for (int i = 0; i < input_tensor.size(); i++) {
    // bias optional
    if (input_tensor[i].defined()) {
      inputTensors.emplace_back(input_tensor[i]);
    }
  }
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensors);
}

SmallVector<NPUTensorDesc, N> _nnpack_spatial_convolution_output(
    const SmallVector<Tensor, N>& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

SmallVector<NPUAttrDesc, N> _nnpack_spatial_convolution_attr(
    IntArrayRef padding,
    IntArrayRef stride) {
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[0], padding[0]};
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[0]};
 
  if (padding.size() != 1) {
    paddings[2] = padding[1];
    paddings[3] = padding[1];
  }
  if (stride.size() != 1) {
    stridesSize[3] = stride[1];
  }

  SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
  string dataFormat = "NCHW";
  int64_t groups = 1;

  NPUAttrDesc npuAttrStrides = NPUAttrDesc("strides", stridesSize);
  NPUAttrDesc npuAttrPads = NPUAttrDesc("pads", paddings);
  NPUAttrDesc npuAttrDilations = NPUAttrDesc("dilations", dilations);
  NPUAttrDesc npuAttrGroups = NPUAttrDesc("groups", groups);
  NPUAttrDesc npuAttrDataFormat = NPUAttrDesc("data_format", dataFormat);

  SmallVector<NPUAttrDesc, N> attrs = {
      npuAttrStrides,
      npuAttrPads,
      npuAttrDilations,
      npuAttrGroups,
      npuAttrDataFormat};

  return attrs;
}

Tensor _nnpack_spatial_convolution_output_npu(
    Tensor& result,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride) {
  auto inputs = _nnpack_spatial_convolution_input({input, weight, bias});
  auto outputs = _nnpack_spatial_convolution_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = _nnpack_spatial_convolution_attr(stride, padding);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Conv2D", inputs, outputs, attrs);
  return result;
}

Tensor _nnpack_spatial_convolution_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride) {
  // calculate the output size
  auto outputSize = nnpack_spatial_convolution_npu_output_size(
      input, weight, padding, stride);
  // construct the output tensor of the NPU
  Tensor result =
      at::empty_with_format(outputSize, input.options(), ACL_FORMAT_NC1HWC0);
  _nnpack_spatial_convolution_output_npu(
      result, input, weight, bias, padding, stride);
  return result;
}

} // namespace native
} // namespace at
