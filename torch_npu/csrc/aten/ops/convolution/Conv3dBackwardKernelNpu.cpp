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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


at::Tensor conv3d_backward_inputmask(at::Tensor &gradInput, const at::Tensor &input,
    const at::Tensor &grad, const at::Tensor &weight,
    at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups) {
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
  at::IntArrayRef inputSize = input.sizes();
  at::Tensor weightCast = weight.to(grad.dtype());

  OpCommand cmd;
  cmd.Name("Conv3DBackpropInput")
    .Input(inputSize, at::kInt)
    .Input(weightCast, "filter", ACL_FORMAT_NCDHW)
    .Input(grad, "out_backprop", ACL_FORMAT_NCDHW)
    .Output(gradInput, "y", ACL_FORMAT_NCDHW)
    .Attr("strides", stridesSize)
    .Attr("pads", paddings)
    .Attr("dilations", dilations)
    .Attr("groups", groups)
    .Attr("data_format", (string) "NCDHW")
    .Run();
  return gradInput;
}

at::Tensor conv3d_backward_weightmask(at::Tensor &gradWeight, const at::Tensor &input,
    const at::Tensor &grad, const at::Tensor &weight,
    at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups) {
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
  at::IntArrayRef inputSize = weight.sizes();

  OpCommand cmd;
  cmd.Name("Conv3DBackpropFilter")
    .Input(input, "x", ACL_FORMAT_NCDHW)
    .Input(inputSize, at::kInt)
    .Input(grad, "out_backprop", ACL_FORMAT_NCDHW)
    .Output(gradWeight, "y", ACL_FORMAT_NCDHW)
    .Attr("strides", stridesSize)
    .Attr("pads", paddings)
    .Attr("dilations", dilations)
    .Attr("groups", groups)
    .Attr("data_format", (string) "NCDHW")
    .Run();

  return gradWeight;
}

at::Tensor conv3d_backward_biasmask(at::Tensor &gradBias, const at::Tensor &input,
    const at::Tensor &grad, const at::Tensor &weight,
    at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, int64_t groups) {
  // constructs the input and output NPUTensorDesc
  if (input.numel() == input.size(0) * input.size(1) * input.size(2)) {
    at::Tensor gradView =
        grad.contiguous().view({grad.size(0), grad.size(1), grad.size(2)});
    at::sum_out(gradBias, gradView, c10::SmallVector<int64_t, N>{0});
  } else {
    at::Tensor gradView =
        grad.contiguous().view({grad.size(0), grad.size(1), grad.size(2), -1});
    at::sum_out(gradBias, gradView, c10::SmallVector<int64_t, N>{0, 2, 3});
  }

  return gradBias;
}

// interface
tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_conv3d_backward(
    const at::Tensor &input, const at::Tensor &grad,
    const at::Tensor &weight, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, 
    int64_t groups, std::array<bool, 3> grad_input_mask) {

  at::Tensor gradInput;
  at::Tensor gradWeight;
  at::Tensor gradBias;

  if (grad_input_mask[0]) {
    // format should be NDC1HWC0
    gradInput = OpPreparation::ApplyTensorWithFormat(
        input.sizes(), input.options(), ACL_FORMAT_NDC1HWC0);

    conv3d_backward_inputmask(
        gradInput, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[1]) {
    // format should be FRACTAL_Z_3D
    gradWeight = OpPreparation::ApplyTensorWithFormat(
        weight.sizes(), weight.options().dtype(at::kFloat), CalcuOpUtil::get_tensor_npu_format(weight));

    conv3d_backward_weightmask(
        gradWeight, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[2]) {
    // format should be NCHW, gradias.size = grad.size(1)
    gradBias = OpPreparation::ApplyTensorWithFormat(
        {grad.size(1)}, grad.options(), ACL_FORMAT_NCHW);

    conv3d_backward_biasmask(
        gradBias, input, grad, weight, stride, padding, dilation, groups);
  }

  return std::make_tuple(gradInput, gradWeight, gradBias);
}

} // namespace native
} // namespace at_npu
