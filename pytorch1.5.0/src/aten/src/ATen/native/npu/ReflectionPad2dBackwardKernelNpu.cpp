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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& reflection_pad2d_backward_out_npu_nocheck(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef padding) {
  TORCH_CHECK(input.scalar_type() != ScalarType::Float,
    "PadV3Grad don't supports torch.float!");      
  SmallVector<int64_t, N> vectorInt;
  SmallVector<int64_t, N> paddingsVector = array_to_small_vector(padding);
  paddingsVector.resize(2 * input.dim(), 0);
  for (int64_t i = paddingsVector.size(); i > 0; i -= 2) {
    vectorInt.emplace_back(paddingsVector[i - 2]);
    vectorInt.emplace_back(paddingsVector[i - 1]);
  } 
  
  OpCommand cmd;
  cmd.Name("PadV3Grad")
    .Input(gradOutput)
    .Input(vectorInt, at::kInt)
    .Output(gradInput)
    .Attr("mode", (string)"reflect")
    .Attr("paddings_contiguous", true)
    .Run();

  return gradInput;
}

Tensor& reflection_pad2d_backward_out_npu(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef padding) {
    OpPreparation::CheckOut(
      {input, gradOutput},
      gradInput,
      input); 

    OpPipeWithDefinedOut pipe;
    return pipe.CheckMemory({input, gradOutput}, {gradInput})
    .Func([&gradOutput, &input, &padding](Tensor& gradInput)
    {reflection_pad2d_backward_out_npu_nocheck(
      gradInput, 
      gradOutput, 
      input, 
      padding);})
    .Call(gradInput); 
}


Tensor reflection_pad2d_backward_npu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef padding) {
  Tensor gradInput = OpPreparation::ApplyTensor(input);
  reflection_pad2d_backward_out_npu_nocheck(
    gradInput, 
    gradOutput, 
    input, 
    padding);
  return gradInput;
}

} // namespace native
} // namespace at
