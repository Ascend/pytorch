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

SmallVector<NPUTensorDesc, N> conv_tbc_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  SmallVector<Tensor, N> inputTensors;
  for (int i = 0; i < inputTensor.size(); i++) {
    if (inputTensor[i].defined()) {
      inputTensors.emplace_back(inputTensor[i]);
    }
  }

  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensors);
}

SmallVector<NPUTensorDesc, N> conv_tbc_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> conv_tbc_npu_attr(int64_t pad) {
  SmallVector<int64_t, N> paddings = {0, 0, pad, pad};
  SmallVector<int64_t, N> stridesSize = {1, 1, 1, 1};
  SmallVector<int64_t, N> dilations = {1, 1, 1, 1};

  string dataFormat = "NCHW";

  NPUAttrDesc npuAttrPads = NPUAttrDesc("pads", paddings);
  NPUAttrDesc npuAttrStrides = NPUAttrDesc("strides", stridesSize);
  NPUAttrDesc npuAttrDilations = NPUAttrDesc("dilations", dilations);
  NPUAttrDesc npuAttrDataFormat = NPUAttrDesc("data_format", dataFormat);

  SmallVector<NPUAttrDesc, N> attrs = {
      npuAttrPads, npuAttrStrides, npuAttrDilations, npuAttrDataFormat};

  return attrs;
}

Tensor& conv_tbc_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    int64_t pad) {
  // constructs the input and output NPUTensorDesc

  auto inputs = conv_tbc_npu_input(
      {self.transpose(0, 2).transpose(0, 1).unsqueeze(2),
       weight.transpose(0, 2).unsqueeze(2),
       bias});

  auto outputs = conv_tbc_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = conv_tbc_npu_attr(pad);
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Conv2D", inputs, outputs, attrs);

  return result;
}

Tensor conv_tbc_npu(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    int64_t pad) {
  // check the shape of input tensors
  TORCH_CHECK(
      self.dim() == 3, "Input must have 3 dims: time, batch, in_channel.");
  TORCH_CHECK(
      weight.dim() == 3,
      "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1-D.");
  TORCH_CHECK(
      self.size(2) == weight.size(1),
      "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tenso.");
  TORCH_CHECK(
      weight.size(2) == bias.size(0),
      "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");

  // calculate the output size
  auto outputSize = conv_tbc_npu_output_size(self, weight, bias, pad);

  // construct the output tensor of the NPU
  Tensor result =
      at::empty_with_format(outputSize, self.options(), ACL_FORMAT_NCHW);

  // calculate the output result of the NPU
  conv_tbc_out_npu(result, self, weight, bias, pad);

  result = result.squeeze(2).transpose(0, 2).transpose(1, 2);
  return result;
}

} // namespace native
} // namespace at
