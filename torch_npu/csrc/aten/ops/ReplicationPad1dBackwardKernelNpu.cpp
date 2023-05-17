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

at::Tensor& NPUNativeFunctions::replication_pad1d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor input_cp = input.unsqueeze(0);
  at::Tensor grad_output_cp = grad_output.unsqueeze(0);
  NPUNativeFunctions::replication_pad2d_backward_out(grad_output_cp, input_cp, paddings, grad_input);
  grad_input.squeeze_(0);
  return grad_input;
}

at::Tensor NPUNativeFunctions::replication_pad1d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor input_cp = input.unsqueeze(0);
  at::Tensor grad_output_cp = grad_output.unsqueeze(0);
  at::Tensor grad_input = NPUNativeFunctions::replication_pad2d_backward(grad_output_cp, input_cp, paddings);
  grad_input.squeeze_(0);
  return grad_input;
}
} // namespace native
} // namespace at_npu
