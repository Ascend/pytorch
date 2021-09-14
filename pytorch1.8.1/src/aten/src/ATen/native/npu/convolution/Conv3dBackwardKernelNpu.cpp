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

#include "ATen/native/npu/utils/OpAdapter.h"
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor conv3d_backward_inputmask(Tensor &gradInput, const Tensor &input,
                                     const Tensor &grad, const Tensor &weight,
                                     IntArrayRef stride, IntArrayRef padding,
                                     IntArrayRef dilation, int64_t groups) {
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
  IntArrayRef inputSize = input.sizes();
  Tensor weightCast = weight.to(grad.dtype());
  
  OpCommand cmd;
  cmd.Name("Conv3DBackpropInput")
    .Input(inputSize, at::kInt)
    .Input(weightCast)
    .Input(grad)
    .Output(gradInput)
    .Attr("strides", stridesSize)
    .Attr("pads", paddings)
    .Attr("dilations", dilations)
    .Attr("groups", groups)
    .Attr("data_format", (string) "NCDHW")
    .Run(); 
  return gradInput;
}

Tensor conv3d_backward_weightmask(Tensor &gradWeight, const Tensor &input,
                                      const Tensor &grad, const Tensor &weight,
                                      IntArrayRef stride, IntArrayRef padding,
                                      IntArrayRef dilation, int64_t groups) {
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
  IntArrayRef inputSize = weight.sizes();

  OpCommand cmd;
  cmd.Name("Conv3DBackpropFilter")
    .Input(input)
    .Input(inputSize, at::kInt)
    .Input(grad)
    .Output(gradWeight)
    .Attr("strides", stridesSize)
    .Attr("pads", paddings)
    .Attr("dilations", dilations)
    .Attr("groups", groups)
    .Attr("data_format", (string) "NCDHW")
    .Run();

  return gradWeight;
}

Tensor conv3d_backward_biasmask(Tensor &gradBias, const Tensor &input,
                                    const Tensor &grad, const Tensor &weight,
                                    IntArrayRef stride, IntArrayRef padding,
                                    IntArrayRef dilation, int64_t groups) {
  // constructs the input and output NPUTensorDesc
  if (input.numel() == input.size(0) * input.size(1) * input.size(2)) {
    Tensor gradView =
        grad.contiguous().view({grad.size(0), grad.size(1), grad.size(2)});
    at::sum_out(gradBias, gradView, SmallVector<int64_t, N>{0});
  } else {
    Tensor gradView =
        grad.contiguous().view({grad.size(0), grad.size(1), grad.size(2), -1});
    at::sum_out(gradBias, gradView, SmallVector<int64_t, N>{0, 2, 3});
  }

  return gradBias;
}

//interface
tuple<Tensor, Tensor, Tensor>
conv3d_backward_npu(const Tensor &input, const Tensor &grad,
                    const Tensor &weight, IntArrayRef stride,
                    IntArrayRef padding, IntArrayRef dilation, int64_t groups,
                    std::array<bool, 3> grad_input_mask) {

  Tensor gradInput;
  Tensor gradWeight;
  Tensor gradBias;
 
  if (grad_input_mask[0]) {
    //format should be NDC1HWC0
    gradInput = at::empty_with_format(
        input.sizes(), input.options(), ACL_FORMAT_NDC1HWC0);
    
    conv3d_backward_inputmask(
        gradInput, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[1]) {
    //format should be FRACTAL_Z_3D
    gradWeight = at::empty_with_format(
        weight.sizes(), weight.options().dtype(kFloat), ACL_FRACTAL_Z_3D);
    
    conv3d_backward_weightmask(
        gradWeight, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[2]) {
    //format should be NCHW, gradias.size = grad.size(1)
    gradBias = at::empty_with_format(
        {grad.size(1)}, grad.options(), ACL_FORMAT_NCHW);
    
    conv3d_backward_biasmask(
        gradBias, input, grad, weight, stride, padding, dilation, groups);
  }

  return std::make_tuple(gradInput, gradWeight, gradBias);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_conv3d_backward", TORCH_FN(conv3d_backward_npu));
}
} // namespace native
} // namespace at
