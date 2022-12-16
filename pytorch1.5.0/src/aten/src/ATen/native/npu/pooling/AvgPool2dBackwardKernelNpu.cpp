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
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& avg_pool2d_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {

  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  if (kernel_size.size() == 1) {
    SmallVector<int64_t, SIZE> kernel_sizes = {kernel_size[0], kernel_size[0]};
    kernel_size = IntArrayRef(kernel_sizes);
  }

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  stride = stride.empty() ? kernel_size : stride;

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  if (padding.size() == 1) {
    SmallVector<int64_t, SIZE> paddings = {padding[0], padding[0]};
    padding = IntArrayRef(paddings);
  }

  const int64_t ndim = self.ndimension();

  TORCH_CHECK((ndim == 3 || ndim == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  // constructs the attr of the NPUAttrDesc
  // required attr
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (!stride.empty()) {
    strideH = stride[0];
    strideW = stride[1];
  }
  SmallVector<int64_t, N> kernelSize = {1, 1, kernel_size[0], kernel_size[1]};
  SmallVector<int64_t, N> stridesSize = {1, 1, strideH, strideW};

  // optional attr
  string padding_mode = "CALCULATED";
  SmallVector<int64_t, N> pads = {padding[0], padding[0], padding[1], padding[1]};
  string format = "NCHW";
  bool pooling = false;
  bool exclusive = (count_include_pad == false) ? true : false;

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("AvgPoolV2Grad")
     .Input(self.sizes())
     .Input(grad_output)
     .Output(grad_input)
     .Attr("ksize", kernelSize)
     .Attr("strides", stridesSize)
     .Attr("padding_mode", padding_mode)
     .Attr("pads", pads)
     .Attr("data_format", format)
     .Attr("global_pooling", pooling)
     .Attr("ceil_mode", ceil_mode)
     .Attr("exclusive", exclusive)
     .Run();

  return grad_input;
}

Tensor avg_pool2d_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor grad_input = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  avg_pool2d_backward_out_npu(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  return grad_input;
}

} // namespace native
} // namespace at
