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

SmallVector<NPUTensorDesc, N> quantized_max_pool2d_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> quantized_max_pool2d_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  auto outputs = CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);

  return outputs;
}

SmallVector<NPUAttrDesc, N> quantized_max_pool2d_npu_attr(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (stride.empty()) {
    strideH = kernel_size[0];
    strideW = kernel_size[1];
  } else {
    strideH = stride[0];
    strideW = stride[1];
  }

  SmallVector<int64_t, N> kernelSize_t = {kernel_size[0], kernel_size[1]};
  SmallVector<int64_t, N> strides_t = {strideH, strideW};
  SmallVector<int64_t, N> paddings_t = {padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations_t = {dilation[0], dilation[0], dilation[1], dilation[1]};

  IntArrayRef kernelSize = IntArrayRef(kernelSize_t);
  IntArrayRef strides = IntArrayRef(strides_t);
  IntArrayRef paddings = IntArrayRef(paddings_t);
  IntArrayRef dilations = IntArrayRef(dilations_t);
  NPUAttrDesc npuAttrKsize = NPUAttrDesc("window", kernelSize);
  NPUAttrDesc npuAttrStrides = NPUAttrDesc("stride", strides);
  NPUAttrDesc npuAttrMode = NPUAttrDesc("mode", (int64_t) 0);
  NPUAttrDesc npuAttrPadding = NPUAttrDesc("pad", paddings);
  NPUAttrDesc npuAttrDilation = NPUAttrDesc("dilation", dilations);
  NPUAttrDesc npuAttrGlobalPooling = NPUAttrDesc("global_pooling", false);
  NPUAttrDesc npuAttrCeilmode = NPUAttrDesc("ceil_mode", (int64_t) !ceil_mode);

  SmallVector<NPUAttrDesc, N> attrs = {npuAttrKsize,
                                       npuAttrStrides,
                                       npuAttrMode,
                                       npuAttrPadding,
                                       npuAttrDilation,
                                       npuAttrGlobalPooling,
                                       npuAttrCeilmode};

  return attrs;
}

Tensor& quantized_max_pool2d_out_npu(
    Tensor& output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // constructs the input and output NPUTensorDesc
  auto inputs = quantized_max_pool2d_npu_input({self});
  auto outputs = quantized_max_pool2d_npu_output({output});

  // constructs the attr of the NPUAttrDesc
  auto attrs = quantized_max_pool2d_npu_attr(
      kernel_size, stride, padding, dilation, ceil_mode);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate(
      "Pooling", inputs, outputs, attrs);

  return output;
}

Tensor quantized_max_pool2d_npu(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // calculate the output size
  auto outputSizes = quantized_max_pool2d_npu_output_size(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  
  // construct the output tensor of the NPU
  Tensor output = at::empty_with_format(
      outputSizes, self.options(), ACL_FORMAT_NC1HWC0);

  // calculate the output result of the NPU
  quantized_max_pool2d_out_npu(
      output, self, kernel_size, stride, padding, dilation, ceil_mode);
  return output;
}

} // namespace native
} // namespace at
