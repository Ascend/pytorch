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
at::Tensor NPUNativeFunctions::conv_tbc(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
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
  int64_t Co = weight.size(2);
  int64_t Wo = (self.size(0) + 2 * pad - (weight.size(0) - 1) - 1) + 1;

  c10::SmallVector<int64_t, SIZE> outputSize = {self.size(1), Co, 1, Wo};

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_NCHW);

  c10::SmallVector<int64_t, N> paddings = {0, 0, pad, pad};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, 1, 1};
  c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1};

  at::Tensor self_tensor = self.transpose(0, 2).transpose(0, 1).unsqueeze(2);
  at::Tensor weight_tensor = weight.transpose(0, 2).unsqueeze(2);

  OpCommand cmd;
  cmd.Name("Conv2D")
    .Input(self_tensor)
    .Input(weight_tensor)
    .Input(bias)
    .Output(result)
    .Attr("pads", paddings)
    .Attr("strides", stridesSize)
    .Attr("dilations", dilations)
    .Attr("data_format", (string)"NCHW")
    .Run();

  result = result.squeeze(2).transpose(0, 2).transpose(1, 2);
  return result;
}
} // namespace native
} // namespace at_npu