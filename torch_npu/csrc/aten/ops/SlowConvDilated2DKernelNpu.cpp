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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::slow_conv_dilated2d(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {

  if (stride[0] == 0) {
    AT_ERROR("slow_conv_dilated2d_npu_output_size: stride[0] can not be zero");
  }
  if (padding[0] < 0 || padding[1] < 0){
    AT_ERROR("slow_conv_dilated2d_npu_output_size: padding can not be less than zero");
  }
  auto outputSize = slow_conv_dilated2d_npu_output_size(
      self, weight, stride, padding, dilation);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, self.options(), ACL_FORMAT_NC1HWC0);
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  int64_t groups = 1;
  c10::SmallVector<int64_t,N> stridesSize = {1,1,stride[0],stride[1]};
  c10::SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};

  OpCommand cmd;
  cmd.Name("Conv2D")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(weight, "filter", ACL_FORMAT_NCHW);
  if (bias.defined()){
     cmd.Input(bias);
  }
  cmd.Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", "NCHW")
      .Run();

  return result;
}

} // namespace native
} // namespace at_npu