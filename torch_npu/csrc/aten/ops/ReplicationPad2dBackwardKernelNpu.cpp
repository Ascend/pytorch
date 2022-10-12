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

at::Tensor& replication_pad2d_backward_out_npu_nocheck(
    at::Tensor& gradInput,
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  c10::SmallVector<int64_t, N> vectorInt;
  c10::SmallVector<int64_t, N> paddingsVector = array_to_small_vector(padding);
  paddingsVector.resize(2 * input.dim(), 0);
  for (int64_t i = paddingsVector.size(); i > 1; i -= 2) {
    vectorInt.emplace_back(paddingsVector[i - 2]);
    vectorInt.emplace_back(paddingsVector[i - 1]);
  }

  OpCommand cmd;
  cmd.Name("PadV3Grad")
      .Input(gradOutput)
      .Input(vectorInt, at::kInt)
      .Output(gradInput)
      .Attr("mode", (string)"edge")
      .Attr("paddings_contiguous", true)
      .Run();

  return gradInput;
}

at::Tensor& NPUNativeFunctions::replication_pad2d_backward_out(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::Tensor& gradInput) {
  OpPreparation::CheckOut(
      {input, gradOutput},
      gradInput,
      input);
  return replication_pad2d_backward_out_npu_nocheck(gradInput, gradOutput, input, padding);
}

at::Tensor NPUNativeFunctions::replication_pad2d_backward(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  at::Tensor gradInput = OpPreparation::ApplyTensor(input);
  replication_pad2d_backward_out(gradOutput, input, padding, gradInput);

  return gradInput;
}
} // namespace native
} // namespace at_npu
