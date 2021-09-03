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

#include <c10/npu/NPUCachingAllocator.h>
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> col2im_backward_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> col2im_backward_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> col2im_backward_npu_attr(
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
    
  SmallVector<int64_t, N> kernelSize = {1, kernel_size[0], kernel_size[1], 1};

  SmallVector<int64_t, N> stridesSize = {1, stride[0], stride[1], 1};

//    SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  string paddings = "SAME";
     
  SmallVector<int64_t, N> dilations = {1, dilation[0], dilation[1], 1};
      
  NPUAttrDesc npuAttrKsize = NPUAttrDesc("ksizes", kernelSize);
  NPUAttrDesc npuAttrStrides = NPUAttrDesc("strides", stridesSize);
  NPUAttrDesc npuAttrPads = NPUAttrDesc("padding", paddings);
  NPUAttrDesc npuAttrDilations = NPUAttrDesc("dilations", dilations);
  SmallVector<NPUAttrDesc, N> attrs = {
      npuAttrKsize,
      npuAttrStrides,
      npuAttrPads,
      npuAttrDilations
  };
 return attrs;
}

Tensor col2im_backward_out_npu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  auto inputs = col2im_backward_npu_input({grad_output});

  auto outputs = col2im_backward_npu_output({grad_input});
     // constructs the attr of the NPUAttrDesc
  auto attrs = col2im_backward_npu_attr(kernel_size, stride, padding, dilation);
  CalcuOpUtil::execute_npu_operate("ExtractImagePatches", inputs, outputs,attrs);
  return grad_input;
}

Tensor& col2im_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  col2im_backward_out_npu_template(grad_input, grad_output,kernel_size, dilation,padding,stride);
  return grad_input;
}

Tensor col2im_backward_npu(
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
    // construct the input tensor of the NPU
  Tensor grad_input = at::empty_with_format(grad_output.sizes(), grad_output.options(),CalcuOpUtil::get_tensor_npu_format(grad_output));

  col2im_backward_out_npu_template(grad_input, grad_output,kernel_size, dilation,padding,stride);

  return grad_input;

}
}
}

