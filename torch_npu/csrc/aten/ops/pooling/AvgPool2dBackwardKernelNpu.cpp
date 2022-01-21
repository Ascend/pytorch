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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& avg_pool2d_backward_out_npu_nocheck(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    at::Tensor& grad_input) {
  at::Tensor orig_input_shape_cpu = at::from_blob((void*)self.sizes().data(), {self.dim()}, at::kLong).to(at::kInt);
  at::Tensor orig_input_shape_npu = CalcuOpUtil::copy_tensor_host_to_device(orig_input_shape_cpu);
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (!stride.empty()) {
    strideH = stride[0];
    strideW = stride[1];
  }
  c10::SmallVector<int64_t, N> kernelSize = {1, 1, kernel_size[0], kernel_size[1]};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, strideH, strideW};
  string padding_mode = "CALCULATED";
  c10::SmallVector<int64_t, N> pads = {padding[0], padding[0], padding[1], padding[1]};
  string format = "NCHW";
  bool pooling = false;
  bool exclusive = (count_include_pad == false) ? true : false;

  OpPreparation::CheckMemory({grad_output, self}, {grad_input});
  OpCommand cmd;
  cmd.Name("AvgPoolV2Grad")
     .InputPair(orig_input_shape_npu, orig_input_shape_cpu)
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

at::Tensor& NPUNativeFunctions::avg_pool2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& grad_input) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  if (kernel_size.size() == 1) {
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {kernel_size[0], kernel_size[0]};
    kernel_size = at::IntArrayRef(kernel_sizes);
  }
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  stride = stride.empty() ? kernel_size : stride;
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  if (padding.size() == 1) {
    c10::SmallVector<int64_t, SIZE> paddings = {padding[0], padding[0]};
    padding = at::IntArrayRef(paddings);
  }
  const int64_t ndim = self.ndimension();
  TORCH_CHECK((ndim == 3 || ndim == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  avg_pool2d_backward_out_npu_nocheck(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      grad_input);
  return grad_input;
}

at::Tensor NPUNativeFunctions::avg_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);

  NPUNativeFunctions::avg_pool2d_backward_out(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      grad_input);
  return grad_input;
}


} // namespace native
} // namespace at_npu
