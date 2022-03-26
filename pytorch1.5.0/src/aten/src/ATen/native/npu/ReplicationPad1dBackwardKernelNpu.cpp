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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;


Tensor& replication_pad1d_backward_out_npu_nocheck(Tensor& gradInput, const Tensor& gradOutput, const Tensor& input, IntArrayRef padding) {
  TORCH_CHECK(input.scalar_type() != ScalarType::Float, "PadV3Grad don't supports torch.float!");
  SmallVector<int64_t, N> vectorInt;
  SmallVector<int64_t, N> paddingsVector = array_to_small_vector(padding);
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

Tensor& replication_pad1d_backward_out_npu(Tensor& gradInput, const Tensor& gradOutput, const Tensor& input, IntArrayRef padding) {
  SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  Tensor inputCopy = input;
  int diff = 4 - inputCopy.dim();
  for (; diff > 0; diff--) {
    inputCopy = inputCopy.unsqueeze(0);
  } 

  Tensor gradOutputCopy = gradOutput;
  int diff1 = 4 - gradOutputCopy.dim();
  for (; diff1 > 0; diff1--) {
    gradOutputCopy = gradOutputCopy.unsqueeze(0);
  } 

  OpPreparation::CheckOut(
      {input, gradOutput},
      gradInput,
      inputCopy);
  replication_pad1d_backward_out_npu_nocheck(gradInput, gradOutputCopy, inputCopy, padding);
  for (; diff > 0; diff--) {
    gradInput = gradInput.squeeze(0);
 }
  return gradInput;
 }

Tensor replication_pad1d_backward_npu(const Tensor& gradOutput, const Tensor& input, IntArrayRef padding) {
  SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  Tensor inputCopy = input;
  int diff = 4 - inputCopy.dim();
  for (; diff > 0; diff--) {
    inputCopy = inputCopy.unsqueeze(0);
  } 

  Tensor gradOutputCopy = gradOutput;
  int diff1 = 4 - gradOutputCopy.dim();
  for (; diff1 > 0; diff1--) {
    gradOutputCopy = gradOutputCopy.unsqueeze(0);
  } 

  Tensor gradInput = OpPreparation::ApplyTensor(inputCopy);

  replication_pad1d_backward_out_npu_nocheck(gradInput, gradOutputCopy, inputCopy, paddings);
  for (; diff > 0; diff--) {
    gradInput = gradInput.squeeze(0);
 }
  return gradInput;
 }
 } // namespace native
 } // namespace at