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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include <ATen/native/Pool.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& NPUNativeFunctions::max_pool2d_with_indices_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& grad_input) {

  int64_t strideH = 1;
  int64_t strideW = 1;
  if (stride.empty()) {
    strideH = kernel_size[0];
    strideW = kernel_size[1];
  } else {
    strideH = stride[0];
    strideW = stride[1];
  }

  c10::SmallVector<int64_t, N> kernelSize = {1, kernel_size[0], kernel_size[1], 1};
  c10::SmallVector<int64_t, N> stridesSize = {1, strideH, strideW, 1};
  c10::SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
  c10::SmallVector<int64_t, N> dilations = {1, dilation[0], dilation[1], 1};
  OpCommand cmd;
  cmd.Name("MaxPoolGradWithArgmaxV1")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(grad_output, "grad", ACL_FORMAT_NCHW)
      .Input(indices, "argmax", ACL_FORMAT_NCHW, "uint16")
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("ksize", kernelSize)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("ceil_mode", ceil_mode)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output_,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
  c10::SmallVector<int64_t, SIZE> kernel_sizes = {kH, kW};
  at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : at::native::safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : at::native::safe_downcast<int, int64_t>(stride[1]);
  c10::SmallVector<int64_t, SIZE> strides = {dH, dW};
  at::IntArrayRef stridess = at::IntArrayRef(strides);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = at::native::safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : at::native::safe_downcast<int, int64_t>(padding[1]);
  c10::SmallVector<int64_t, SIZE> paddings = {padH, padW};
  at::IntArrayRef padss = at::IntArrayRef(paddings);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = at::native::safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : at::native::safe_downcast<int, int64_t>(dilation[1]);
  c10::SmallVector<int64_t, SIZE> dilations = {dilationH, dilationW};
  at::IntArrayRef dilationss = at::IntArrayRef(dilations);

  TORCH_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* get contiguous gradOutput */
  const at::Tensor grad_output = grad_output_.contiguous();

  /* sizes */
  const int64_t inputHeight = self.size(-2);
  const int64_t inputWidth = self.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check = at::native::pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check = at::native::pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  // construct the output tensor of the NPU
  at::Tensor grad_input =  OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  NPUNativeFunctions::max_pool2d_with_indices_backward_out(
      grad_output,
      self,
      kernel_sizess,
      stridess,
      padss,
      dilationss,
      ceil_mode,
      indices,
      grad_input);

  return grad_input;
}


} // namespace native
} // namespace at_npu
