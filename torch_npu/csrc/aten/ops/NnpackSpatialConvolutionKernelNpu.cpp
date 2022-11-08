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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<NPUTensorDesc, N> _nnpack_spatial_convolution_input(
    const c10::SmallVector<at::Tensor, N>& input_tensor) {
  c10::SmallVector<at::Tensor, N> inputTensors;
  for (int i = 0; i < input_tensor.size(); i++) {
    if (input_tensor[i].defined()) {
      inputTensors.emplace_back(input_tensor[i]);
    }
  }
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensors);
}

c10::SmallVector<NPUTensorDesc, N> _nnpack_spatial_convolution_output(
    const c10::SmallVector<at::Tensor, N>& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

c10::SmallVector<NPUAttrDesc, N> _nnpack_spatial_convolution_attr(
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  c10::SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[0], padding[0]};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[0]};

  if (padding.size() != 1) {
    paddings[2] = padding[1];
    paddings[3] = padding[1];
  }
  if (stride.size() != 1) {
    stridesSize[3] = stride[1];
  }

  c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
  string dataFormat = "NCHW";
  int64_t groups = 1;

  NPUAttrDesc npuAttrStrides = NPUAttrDesc("strides", stridesSize);
  NPUAttrDesc npuAttrPads = NPUAttrDesc("pads", paddings);
  NPUAttrDesc npuAttrDilations = NPUAttrDesc("dilations", dilations);
  NPUAttrDesc npuAttrGroups = NPUAttrDesc("groups", groups);
  NPUAttrDesc npuAttrDataFormat = NPUAttrDesc("data_format", dataFormat);

  c10::SmallVector<NPUAttrDesc, N> attrs = {
      npuAttrStrides,
      npuAttrPads,
      npuAttrDilations,
      npuAttrGroups,
      npuAttrDataFormat};

  return attrs;
}

at::Tensor _nnpack_spatial_convolution_output_npu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& result) {
  auto inputs = _nnpack_spatial_convolution_input({input, weight, bias});
  auto outputs = _nnpack_spatial_convolution_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = _nnpack_spatial_convolution_attr(stride, padding);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Conv2D", inputs, outputs, attrs);
  return result;
}

at::Tensor NPUNativeFunctions::_nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = nnpack_spatial_convolution_npu_output_size(
      input, weight, padding, stride);
  auto result_format = CalcuOpUtil::judge_and_get_format_from_input(
      CalcuOpUtil::get_tensor_npu_format(weight) == ACL_FORMAT_FRACTAL_Z,
      input, ACL_FORMAT_NC1HWC0);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, input.options(), result_format);
  _nnpack_spatial_convolution_output_npu(
      input, weight, bias, padding, stride, result);
  return result;
}

} // namespace native
} // namespace at_npu