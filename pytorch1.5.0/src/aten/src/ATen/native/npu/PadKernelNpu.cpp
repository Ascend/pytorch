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

Tensor& pad_out_npu(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddings) {
  SmallVector<int64_t, N> paddingsVector = array_to_small_vector(paddings);
  paddingsVector.resize(2 * input.dim(), 0);

  OpCommand cmd;
  cmd.Name("Pad")
      .Input(input)
      .Input(paddingsVector)
      .Output(output)
      .Run();
  return output;
}

Tensor pad_npu(const Tensor& input, IntArrayRef paddings) {
  auto outputSize = pad_npu_output_size(input, paddings);
  Tensor output = OpPreparation::ApplyTensor(input, outputSize);
  pad_out_npu(output, input, paddings);
  return output;
}

} // namespace native
} // namespace at
